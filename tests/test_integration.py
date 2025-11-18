"""Integration tests for complete imputation workflows."""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

from imputation_showcase.imputation_methods import (
    MeanImputer,
    MedianImputer,
    KNNImputerMethod,
    MICEImputer,
    rmse,
    mae,
)


def test_full_pipeline_diabetes_dataset() -> None:
    """Test a complete imputation and evaluation workflow."""
    # Load real dataset
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame.iloc[:100, :5]  # Subset for speed

    # Keep original for comparison
    original = df.copy()

    # Introduce missing values
    rng = np.random.RandomState(42)
    mask = rng.rand(*df.shape) < 0.1
    df[mask] = np.nan

    # Test multiple imputers
    imputers = [
        MeanImputer(),
        MedianImputer(),
        KNNImputerMethod(k=5),
        MICEImputer(random_state=42),
    ]

    for imputer in imputers:
        # Impute
        imputed = imputer.impute(df)

        # Verify no NaN values remain
        assert not imputed.isna().any().any(), (
            f"{imputer.__class__.__name__} left NaN values"
        )

        # Verify shape preserved
        assert imputed.shape == df.shape, (
            f"{imputer.__class__.__name__} changed shape"
        )

        # Verify index preserved
        pd.testing.assert_index_equal(imputed.index, df.index)

        # Verify columns preserved
        pd.testing.assert_index_equal(imputed.columns, df.columns)

        # Evaluate imputation quality (RMSE should be reasonable)
        for col in df.columns:
            error = rmse(original[col], imputed[col])
            # Error should be less than 5x the column's std dev
            assert error < 5 * original[col].std(), (
                f"{imputer.__class__.__name__} has high error on {col}"
            )


def test_sequential_imputation() -> None:
    """Test that imputers can be chained together."""
    df = pd.DataFrame({
        "a": [1, 2, np.nan, 4, 5],
        "b": [6, np.nan, 8, 9, 10],
        "c": [11, 12, 13, np.nan, 15],
    })

    # First pass: mean imputation
    step1 = MeanImputer().impute(df)
    assert not step1.isna().any().any()

    # Second pass: KNN on already-imputed data (should not change)
    step2 = KNNImputerMethod(k=2).impute(step1)
    assert not step2.isna().any().any()

    # Results should be close (since no NaNs in step1)
    pd.testing.assert_frame_equal(step1, step2, check_exact=False, atol=1e-5)


def test_data_type_preservation() -> None:
    """Test that imputers preserve numeric data types."""
    df = pd.DataFrame({
        "int_col": pd.Series([1, 2, pd.NA, 4], dtype="Int64"),
        "float_col": pd.Series([1.5, np.nan, 3.5, 4.5], dtype="float64"),
    })

    # Convert to float for imputation (required by our imputers)
    df_float = df.astype("float64")
    imputed = MeanImputer().impute(df_float)

    # Verify types are preserved
    assert imputed["int_col"].dtype == np.float64
    assert imputed["float_col"].dtype == np.float64

    # Verify no NaN values
    assert not imputed.isna().any().any()


def test_evaluation_metrics_consistency() -> None:
    """Test that RMSE and MAE metrics work correctly."""
    true_values = pd.Series([1, 2, 3, 4, 5])
    predicted = pd.Series([1.1, 2.2, 2.8, 4.1, 5.0])

    # Calculate metrics
    rmse_val = rmse(true_values, predicted)
    mae_val = mae(true_values, predicted)

    # RMSE should be >= MAE (by Jensen's inequality)
    assert rmse_val >= mae_val, "RMSE should be >= MAE"

    # Both should be positive
    assert rmse_val > 0
    assert mae_val > 0

    # Both should be reasonable (< 1 for this data)
    assert rmse_val < 1.0
    assert mae_val < 1.0


def test_empty_dataframe_handling() -> None:
    """Test that imputers handle edge cases gracefully."""
    # Empty dataframe
    df_empty = pd.DataFrame()

    # Should handle empty dataframe
    imputed = MeanImputer().impute(df_empty)
    assert imputed.shape == (0, 0)


def test_large_proportion_missing() -> None:
    """Test imputers with high percentage of missing data."""
    df = pd.DataFrame({
        "a": [1, np.nan, np.nan, np.nan, 5],
        "b": [np.nan, 2, np.nan, np.nan, 6],
    })

    # 60% missing data - should still work
    imputers = [
        MeanImputer(),
        MedianImputer(),
        KNNImputerMethod(k=1),
    ]

    for imputer in imputers:
        imputed = imputer.impute(df)
        assert not imputed.isna().any().any(), (
            f"{imputer.__class__.__name__} failed with high missingness"
        )


def test_reproducibility_with_random_state() -> None:
    """Test that random_state ensures reproducible results."""
    df = pd.DataFrame({
        "a": [1, 2, np.nan, 4, 5],
        "b": [6, np.nan, 8, 9, 10],
    })

    # Run twice with same random state
    imputer1 = MICEImputer(random_state=42)
    imputer2 = MICEImputer(random_state=42)

    result1 = imputer1.impute(df)
    result2 = imputer2.impute(df)

    # Results should be identical
    pd.testing.assert_frame_equal(result1, result2)


def test_comparison_across_imputers() -> None:
    """Test that different imputers produce different but valid results."""
    df = pd.DataFrame({
        "a": [1, 2, np.nan, 4, 5],
        "b": [6, 7, 8, np.nan, 10],
    })

    mean_result = MeanImputer().impute(df)
    median_result = MedianImputer().impute(df)

    # Both should have no NaN
    assert not mean_result.isna().any().any()
    assert not median_result.isna().any().any()

    # Results may differ (mean != median for skewed data)
    # But both should be valid numeric dataframes
    assert mean_result.shape == df.shape
    assert median_result.shape == df.shape
