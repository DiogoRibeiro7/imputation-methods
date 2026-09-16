import numpy as np
import pandas as pd
import pytest

from imputation_methods import (
    AutoencoderImputer,
    BayesianPCAImputer,
    GAINImputer,
    GaussianProcessImputer,
    HotDeckImputer,
    KNNImputerMethod,
    LOCFImputer,
    MeanImputer,
    MedianImputer,
    MICEImputer,
    MissForestImputer,
    NOCBImputer,
    PMMImputer,
    RegressionImputer,
    SoftImputeImputer,
    StochasticRegressionImputer,
    mae,
    rmse,
)


def test_mean_impute_basic() -> None:
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, np.nan, 7, 8]})
    imputed = MeanImputer().impute(df)
    assert not imputed.isna().any().any()
    assert np.isclose(imputed.loc[2, "a"], (1 + 2 + 4) / 3)


def test_median_impute_basic() -> None:
    df = pd.DataFrame({"a": [1, 2, np.nan, 10]})
    imputed = MedianImputer().impute(df)
    assert not imputed.isna().any().any()
    assert imputed.loc[2, "a"] == 2


def test_knn_impute_basic() -> None:
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, np.nan, 7, 8]})
    imputed = KNNImputerMethod(k=2).impute(df)
    assert not imputed.isna().any().any()


def test_pmm_impute_basic() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, np.nan], "y": [5, 6, 7, 8, 9]})
    imputed = PMMImputer(k=2, random_state=0).impute(df)
    assert not imputed.isna().any().any()


def test_mice_impute_basic() -> None:
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, np.nan, 7, 8]})
    imputed = MICEImputer(random_state=0).impute(df)
    assert not imputed.isna().any().any()


def test_non_numeric_raises() -> None:
    df = pd.DataFrame({"a": [1, 2, np.nan], "b": ["x", "y", "z"]})
    with pytest.raises(TypeError):
        MeanImputer().impute(df)


def test_no_missing() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    imputed = MeanImputer().impute(df)
    pd.testing.assert_frame_equal(imputed, df)


def test_all_missing() -> None:
    df = pd.DataFrame({"a": [np.nan, np.nan]})
    imputed = MeanImputer().impute(df)
    assert imputed.isna().all().all()


def test_single_column() -> None:
    df = pd.DataFrame({"a": [1, np.nan, 3]})
    imputed = MedianImputer().impute(df)
    assert not imputed.isna().any().any()


def test_regression_imputer() -> None:
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, 6, 7, 8]})
    imputed = RegressionImputer().impute(df)
    assert not imputed.isna().any().any()


def test_stochastic_regression_imputer() -> None:
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, 6, 7, 8]})
    imputed = StochasticRegressionImputer(random_state=0).impute(df)
    assert not imputed.isna().any().any()


def test_locf_nocb() -> None:
    df = pd.DataFrame({"a": [np.nan, 1, np.nan, 3, np.nan]})
    locf = LOCFImputer().impute(df)
    assert pd.isna(locf.loc[0, "a"])
    assert locf.loc[2, "a"] == 1
    nocb = NOCBImputer().impute(df)
    assert nocb.loc[0, "a"] == 1
    assert nocb.loc[2, "a"] == 3


def test_hot_deck_imputer() -> None:
    df = pd.DataFrame({"group": [0, 0, 1, 1], "a": [1.0, np.nan, 3.0, np.nan]})
    imputed = HotDeckImputer(
        stratify_cols=["group"],
        random_state=0,
    ).impute(df)
    assert not imputed["a"].isna().any()


def test_miss_forest_imputer() -> None:
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, 6, 7, np.nan]})
    imputed = MissForestImputer(random_state=0).impute(df)
    assert not imputed.isna().any().any()


def test_soft_impute() -> None:
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, np.nan]})
    imputed = SoftImputeImputer().impute(df)
    assert not imputed.isna().any().any()


def test_bayesian_pca_imputer() -> None:
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, np.nan]})
    imputed = BayesianPCAImputer().impute(df)
    assert not imputed.isna().any().any()


def test_autoencoder_imputer() -> None:
    """Test autoencoder imputer with sufficient iterations to converge."""
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, np.nan]})
    imputed = AutoencoderImputer(random_state=0, max_iter=500).impute(df)
    assert not imputed.isna().any().any()


def test_gain_imputer() -> None:
    df = pd.DataFrame({"a": [1, 2, np.nan], "b": [4, np.nan, 6]})
    imputed = GAINImputer(random_state=0).impute(df)
    assert not imputed.isna().any().any()


def test_gaussian_process_imputer() -> None:
    df = pd.DataFrame({"a": [1, 2, np.nan], "b": [4, 5, 6]})
    imputed = GaussianProcessImputer(random_state=0).impute(df)
    assert not imputed.isna().any().any()


# Edge case tests for improved coverage


def test_regression_imputer_single_column() -> None:
    """Test regression imputer with single column (no predictors)."""
    df = pd.DataFrame({"a": [1, 2, np.nan, 4]})
    imputed = RegressionImputer().impute(df)
    # Should leave NaN as is when no predictors available
    assert imputed.isna().any().any()


def test_stochastic_regression_single_column() -> None:
    """Test stochastic regression imputer with single column."""
    df = pd.DataFrame({"a": [1, 2, np.nan, 4]})
    imputed = StochasticRegressionImputer(random_state=0).impute(df)
    # Should leave NaN as is when no predictors available
    assert imputed.isna().any().any()


def test_gaussian_process_single_column() -> None:
    """Test Gaussian process imputer with single column."""
    df = pd.DataFrame({"a": [1, 2, np.nan, 4]})
    imputed = GaussianProcessImputer(random_state=0).impute(df)
    # Should leave NaN as is when no predictors available
    assert imputed.isna().any().any()


def test_pmm_imputer_no_predictors() -> None:
    """Test PMM imputer when no predictors are available."""
    df = pd.DataFrame({"a": [1, 2, np.nan, 4]})
    imputed = PMMImputer(k=2, random_state=0).impute(df)
    # Should fall back to mean imputation
    assert not imputed.isna().any().any()
    assert np.isclose(imputed.loc[2, "a"], (1 + 2 + 4) / 3)


def test_pmm_imputer_empty_observed() -> None:
    """Test PMM when observed data is empty for predictors."""
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1, 2, 3]})
    imputed = PMMImputer(k=2, random_state=0).impute(df)
    # Should handle gracefully
    assert imputed.isna().all()["a"]


def test_bayesian_pca_single_column() -> None:
    """Test Bayesian PCA with single column (cannot reduce dimensions)."""
    df = pd.DataFrame({"a": [1, 2, np.nan, 4]})
    imputed = BayesianPCAImputer().impute(df)
    # Should return copy of original when only one column
    pd.testing.assert_frame_equal(imputed, df)


def test_hot_deck_imputer_no_stratification() -> None:
    """Test hot deck imputer without stratification columns."""
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, np.nan, 7, 8]})
    imputed = HotDeckImputer(random_state=0).impute(df)
    assert not imputed.isna().any().any()


def test_hot_deck_empty_donors_in_group() -> None:
    """Test hot deck when a group has no donors."""
    df = pd.DataFrame({"group": [0, 0, 1, 1], "a": [np.nan, np.nan, 3.0, 4.0]})
    imputer = HotDeckImputer(stratify_cols=["group"], random_state=0)
    imputed = imputer.impute(df)
    # Should fall back to full column donors when group has none
    assert not imputed["a"].isna().any()


def test_locf_all_nan() -> None:
    """Test LOCF with all NaN values."""
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
    imputed = LOCFImputer().impute(df)
    assert imputed.isna().all().all()


def test_nocb_all_nan() -> None:
    """Test NOCB with all NaN values."""
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
    imputed = NOCBImputer().impute(df)
    assert imputed.isna().all().all()


def test_locf_trailing_nan() -> None:
    """Test LOCF with trailing NaN values."""
    df = pd.DataFrame({"a": [1, 2, np.nan, np.nan]})
    imputed = LOCFImputer().impute(df)
    assert imputed.loc[2, "a"] == 2
    assert imputed.loc[3, "a"] == 2


def test_nocb_leading_nan() -> None:
    """Test NOCB with leading NaN values."""
    df = pd.DataFrame({"a": [np.nan, np.nan, 3, 4]})
    imputed = NOCBImputer().impute(df)
    assert imputed.loc[0, "a"] == 3
    assert imputed.loc[1, "a"] == 3


def test_rmse_calculation() -> None:
    """Test RMSE evaluation metric."""
    true = pd.Series([1, 2, 3, 4, 5])
    pred = pd.Series([1.1, 2.1, 2.9, 4.2, 4.8])
    error = rmse(true, pred)
    expected = np.sqrt(np.mean([0.01, 0.01, 0.01, 0.04, 0.04]))
    assert np.isclose(error, expected)


def test_mae_calculation() -> None:
    """Test MAE evaluation metric."""
    true = pd.Series([1, 2, 3, 4, 5])
    pred = pd.Series([1.1, 2.1, 2.9, 4.2, 4.8])
    error = mae(true, pred)
    expected = np.mean([0.1, 0.1, 0.1, 0.2, 0.2])
    assert np.isclose(error, expected)


def test_median_imputer_even_count() -> None:
    """Test median imputation with even number of values."""
    df = pd.DataFrame({"a": [1, 2, np.nan, 4]})
    imputed = MedianImputer().impute(df)
    # Median of [1, 2, 4] should be 2
    assert imputed.loc[2, "a"] == 2


def test_knn_imputer_k_variation() -> None:
    """Test KNN imputer with different k values."""
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, 6, 7, 8]})
    imputed_k1 = KNNImputerMethod(k=1).impute(df)
    imputed_k3 = KNNImputerMethod(k=3).impute(df)
    assert not imputed_k1.isna().any().any()
    assert not imputed_k3.isna().any().any()
    # Different k values may produce different results
    # Just verify both complete the imputation


def test_multiple_imputers_preserve_index() -> None:
    """Test that imputers preserve DataFrame index."""
    df = pd.DataFrame(
        {"a": [1, 2, np.nan, 4], "b": [5, np.nan, 7, 8]}, index=["w", "x", "y", "z"]
    )
    imputers = [
        MeanImputer(),
        MedianImputer(),
        KNNImputerMethod(k=2),
    ]
    for imputer in imputers:
        imputed = imputer.impute(df)
        pd.testing.assert_index_equal(imputed.index, df.index)


def test_multiple_imputers_preserve_columns() -> None:
    """Test that imputers preserve DataFrame column names."""
    df = pd.DataFrame({"col_a": [1, 2, np.nan], "col_b": [4, np.nan, 6]})
    # Test with imputers that handle multiple missing values gracefully
    imputers = [
        MeanImputer(),
        MedianImputer(),
        KNNImputerMethod(k=1),
        MICEImputer(random_state=0),
    ]
    for imputer in imputers:
        imputed = imputer.impute(df)
        pd.testing.assert_index_equal(imputed.columns, df.columns)


def test_soft_impute_different_init_methods() -> None:
    """Test SoftImpute with different initialization methods."""
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, np.nan]})
    imputed_zero = SoftImputeImputer(init_fill_method="zero").impute(df)
    imputed_mean = SoftImputeImputer(init_fill_method="mean").impute(df)
    assert not imputed_zero.isna().any().any()
    assert not imputed_mean.isna().any().any()


def test_bayesian_pca_with_n_components() -> None:
    """Test Bayesian PCA with explicit number of components."""
    df = pd.DataFrame(
        {"a": [1, np.nan, 3, 4], "b": [4, 5, np.nan, 7], "c": [7, 8, 9, np.nan]}
    )
    imputed = BayesianPCAImputer(n_components=2).impute(df)
    assert not imputed.isna().any().any()


def test_autoencoder_different_architectures() -> None:
    """Test autoencoder with different hidden layer configurations."""
    df = pd.DataFrame({"a": [1, np.nan, 3, 4], "b": [4, 5, np.nan, 7]})
    imputed_small = AutoencoderImputer(
        hidden_layer_sizes=(5,), max_iter=500, random_state=0
    ).impute(df)
    imputed_large = AutoencoderImputer(
        hidden_layer_sizes=(10, 5), max_iter=500, random_state=0
    ).impute(df)
    assert not imputed_small.isna().any().any()
    assert not imputed_large.isna().any().any()


# Input validation tests


def test_knn_invalid_k_type() -> None:
    """Test KNN imputer rejects non-integer k values."""
    with pytest.raises(TypeError, match="k must be an integer"):
        KNNImputerMethod(k=2.5)  # type: ignore


def test_knn_invalid_k_value() -> None:
    """Test KNN imputer rejects non-positive k values."""
    with pytest.raises(ValueError, match="k must be positive"):
        KNNImputerMethod(k=0)
    with pytest.raises(ValueError, match="k must be positive"):
        KNNImputerMethod(k=-1)


def test_pmm_invalid_k_type() -> None:
    """Test PMM imputer rejects non-integer k values."""
    with pytest.raises(TypeError, match="k must be an integer"):
        PMMImputer(k=3.5)  # type: ignore


def test_pmm_invalid_k_value() -> None:
    """Test PMM imputer rejects non-positive k values."""
    with pytest.raises(ValueError, match="k must be positive"):
        PMMImputer(k=0)
    with pytest.raises(ValueError, match="k must be positive"):
        PMMImputer(k=-2)
