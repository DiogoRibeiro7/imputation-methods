"""Tests for Phase 1-4 imputation methods (10 new methods)."""

import numpy as np
import pandas as pd
import pytest

from imputation_methods import (
    ColdDeckImputer,
    ConstantImputer,
    # Phase 2: Distribution-Based
    EndOfDistributionImputer,
    GroupMeanImputer,
    HybridImputer,
    InterpolationImputer,
    # Phase 4: Hybrid & Advanced
    KalmanFilterImputer,
    LinearTrendImputer,
    # Needed for hybrid tests
    MeanImputer,
    # Phase 1: Basic Statistical
    ModeImputer,
    PolynomialTrendImputer,
    # Phase 3: Time Series Advanced
    WeightedMovingAverageImputer,
)

# ============================================================================
# PHASE 1: Basic Statistical Methods
# ============================================================================


class TestModeImputer:
    """Tests for ModeImputer."""

    def test_mode_imputation(self):
        """Test basic mode imputation."""
        df = pd.DataFrame({"a": [1, 2, 2, np.nan, 2, 3]})
        imputer = ModeImputer()
        result = imputer.impute(df)

        assert not result.isna().any().any()
        assert result["a"].iloc[3] == 2.0  # Most frequent value

    def test_mode_with_multiple_modes(self):
        """Test mode when multiple modes exist (takes first)."""
        df = pd.DataFrame({"a": [1, 1, 2, 2, np.nan]})
        imputer = ModeImputer()
        result = imputer.impute(df)

        assert not result.isna().any().any()
        # Should use first mode
        assert result["a"].iloc[4] in [1.0, 2.0]

    def test_mode_with_all_unique(self):
        """Test mode when all values are unique (takes first mode)."""
        df = pd.DataFrame({"a": [1.1, 2.2, 3.3, np.nan]})
        imputer = ModeImputer()
        result = imputer.impute(df)

        assert not result.isna().any().any()
        # When all values unique, mode() returns all, takes first
        assert result["a"].iloc[3] in [1.1, 2.2, 3.3]

    def test_mode_dropna_parameter(self):
        """Test dropna parameter."""
        df = pd.DataFrame({"a": [1, 2, 2, np.nan, 2]})
        imputer = ModeImputer(dropna=True)
        result = imputer.impute(df)

        assert not result.isna().any().any()


class TestConstantImputer:
    """Tests for ConstantImputer."""

    def test_constant_scalar(self):
        """Test constant imputation with scalar value."""
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, 3]})
        imputer = ConstantImputer(fill_value=-999)
        result = imputer.impute(df)

        assert not result.isna().any().any()
        assert result["a"].iloc[1] == -999
        assert result["b"].iloc[0] == -999

    def test_constant_dict(self):
        """Test constant imputation with dictionary."""
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, 3]})
        imputer = ConstantImputer(fill_value={"a": 0, "b": 100})
        result = imputer.impute(df)

        assert not result.isna().any().any()
        assert result["a"].iloc[1] == 0
        assert result["b"].iloc[0] == 100

    def test_constant_unknown_column_raises(self):
        """Test that unknown columns in dict raise error."""
        df = pd.DataFrame({"a": [1, np.nan, 3]})
        imputer = ConstantImputer(fill_value={"a": 0, "z": 100})

        with pytest.raises(ValueError, match="unknown columns"):
            imputer.impute(df)

    def test_constant_default_zero(self):
        """Test default fill value of 0."""
        df = pd.DataFrame({"a": [1, np.nan, 3]})
        imputer = ConstantImputer()
        result = imputer.impute(df)

        assert result["a"].iloc[1] == 0


# ============================================================================
# PHASE 2: Distribution-Based Methods
# ============================================================================


class TestEndOfDistributionImputer:
    """Tests for EndOfDistributionImputer."""

    def test_high_position(self):
        """Test imputation at high end of distribution."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, np.nan]})
        imputer = EndOfDistributionImputer(position="high", k=1)
        result = imputer.impute(df)

        assert not result.isna().any().any()
        mean = df["a"].mean()
        std = df["a"].std()
        expected = mean + 1 * std
        assert result["a"].iloc[5] == pytest.approx(expected, rel=1e-6)

    def test_low_position(self):
        """Test imputation at low end of distribution."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, np.nan]})
        imputer = EndOfDistributionImputer(position="low", k=2)
        result = imputer.impute(df)

        assert not result.isna().any().any()
        mean = df["a"].mean()
        std = df["a"].std()
        expected = mean - 2 * std
        assert result["a"].iloc[5] == pytest.approx(expected, rel=1e-6)

    def test_invalid_position_raises(self):
        """Test that invalid position raises error."""
        with pytest.raises(ValueError, match="position must be"):
            EndOfDistributionImputer(position="middle")


class TestGroupMeanImputer:
    """Tests for GroupMeanImputer."""

    def test_group_mean_basic(self):
        """Test basic group mean imputation."""
        df = pd.DataFrame(
            {"category": [1, 1, 1, 2, 2, 2], "value": [10, np.nan, 12, 20, np.nan, 24]}
        )
        imputer = GroupMeanImputer(group_col="category", method="mean")
        result = imputer.impute(df)

        assert not result["value"].isna().any()
        # Group 1 mean: (10+12)/2 = 11
        assert result["value"].iloc[1] == pytest.approx(11.0)
        # Group 2 mean: (20+24)/2 = 22
        assert result["value"].iloc[4] == pytest.approx(22.0)

    def test_group_median(self):
        """Test group median imputation."""
        df = pd.DataFrame(
            {"category": [1, 1, 1, 2, 2, 2], "value": [10, np.nan, 14, 20, np.nan, 24]}
        )
        imputer = GroupMeanImputer(group_col="category", method="median")
        result = imputer.impute(df)

        assert not result["value"].isna().any()
        # Group 1 median: 12
        assert result["value"].iloc[1] == pytest.approx(12.0)

    def test_group_missing_column_raises(self):
        """Test that missing group column raises error."""
        df = pd.DataFrame({"value": [10, np.nan, 12]})
        imputer = GroupMeanImputer(group_col="category")

        with pytest.raises(ValueError, match="not found in dataframe"):
            imputer.impute(df)

    def test_global_fallback(self):
        """Test global fallback when group has no data."""
        df = pd.DataFrame({"category": [1, 1, 2], "value": [10, 12, np.nan]})
        imputer = GroupMeanImputer(group_col="category", global_fallback=True)
        result = imputer.impute(df)

        # Group 2 has no observed values, should use global mean
        assert not result["value"].isna().any()


# ============================================================================
# PHASE 3: Time Series Advanced Methods
# ============================================================================


class TestWeightedMovingAverageImputer:
    """Tests for WeightedMovingAverageImputer."""

    def test_ewma_basic(self):
        """Test basic EWMA imputation."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, np.nan, 6]})
        imputer = WeightedMovingAverageImputer(alpha=0.5)
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_alpha_validation(self):
        """Test alpha parameter validation."""
        with pytest.raises(ValueError, match="alpha must be in"):
            WeightedMovingAverageImputer(alpha=0)

        with pytest.raises(ValueError, match="alpha must be in"):
            WeightedMovingAverageImputer(alpha=1.5)

    def test_alpha_effects(self):
        """Test that different alpha values produce different results."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 10]})

        result1 = WeightedMovingAverageImputer(alpha=0.1).impute(df)
        result2 = WeightedMovingAverageImputer(alpha=0.9).impute(df)

        # Different alphas should produce different imputed values
        assert result1["a"].iloc[2] != result2["a"].iloc[2]


class TestLinearTrendImputer:
    """Tests for LinearTrendImputer."""

    def test_linear_trend_basic(self):
        """Test basic linear trend imputation."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, np.nan, 6]})
        imputer = LinearTrendImputer()
        result = imputer.impute(df)

        assert not result.isna().any().any()
        # Linear trend should give approximately 3 and 5
        assert result["a"].iloc[2] == pytest.approx(3.0, abs=0.1)
        assert result["a"].iloc[4] == pytest.approx(5.0, abs=0.1)

    def test_single_value_fallback(self):
        """Test fallback when only one observed value."""
        df = pd.DataFrame({"a": [np.nan, 5, np.nan]})
        imputer = LinearTrendImputer()
        result = imputer.impute(df)

        assert not result.isna().any().any()
        # Should use constant (the single value)
        assert all(result["a"] == 5.0)

    def test_use_index_parameter(self):
        """Test use_index parameter."""
        df = pd.DataFrame({"a": [1, np.nan, 3]}, index=[0, 5, 10])
        imputer = LinearTrendImputer(use_index=True)
        result = imputer.impute(df)

        assert not result.isna().any().any()


class TestPolynomialTrendImputer:
    """Tests for PolynomialTrendImputer."""

    def test_polynomial_degree_2(self):
        """Test quadratic polynomial imputation."""
        df = pd.DataFrame({"a": [1, 4, np.nan, 16, np.nan, 36]})
        imputer = PolynomialTrendImputer(degree=2)
        result = imputer.impute(df)

        assert not result.isna().any().any()
        # Should approximate x^2 pattern
        assert result["a"].iloc[2] == pytest.approx(9.0, rel=0.1)

    def test_degree_validation(self):
        """Test degree parameter validation."""
        with pytest.raises(ValueError, match="degree must be >= 1"):
            PolynomialTrendImputer(degree=0)

    def test_degree_adjustment(self):
        """Test automatic degree adjustment for few points."""
        df = pd.DataFrame({"a": [1, np.nan, 3]})
        # Only 2 observed points, can't fit degree 3 polynomial
        imputer = PolynomialTrendImputer(degree=3)
        result = imputer.impute(df)

        assert not result.isna().any().any()


# ============================================================================
# PHASE 4: Hybrid & Advanced Methods
# ============================================================================


class TestKalmanFilterImputer:
    """Tests for KalmanFilterImputer."""

    def test_kalman_basic(self):
        """Test basic Kalman filter imputation."""
        df = pd.DataFrame({"a": [1, np.nan, 3, np.nan, 5]})
        imputer = KalmanFilterImputer()
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_kalman_parameters(self):
        """Test Kalman filter with different parameters."""
        df = pd.DataFrame({"a": [1, np.nan, 3, np.nan, 5]})
        imputer = KalmanFilterImputer(
            process_variance=0.1, measurement_variance=0.1, initial_covariance=0.5
        )
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_kalman_with_initial_state(self):
        """Test Kalman filter with specified initial state."""
        df = pd.DataFrame({"a": [np.nan, 2, np.nan, 4]})
        imputer = KalmanFilterImputer(initial_state=1.0)
        result = imputer.impute(df)

        assert not result.isna().any().any()


class TestColdDeckImputer:
    """Tests for ColdDeckImputer."""

    def test_cold_deck_with_dict(self):
        """Test cold deck with dictionary reference."""
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, 3]})
        imputer = ColdDeckImputer(reference_values={"a": 2.5, "b": 2.0})
        result = imputer.impute(df)

        assert not result.isna().any().any()
        assert result["a"].iloc[1] == 2.5
        assert result["b"].iloc[0] == 2.0

    def test_cold_deck_with_dataframe(self):
        """Test cold deck with DataFrame reference."""
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, 2, 3]})
        ref_df = pd.DataFrame({"a": [10, 20, 30], "b": [100, 200, 300]})
        imputer = ColdDeckImputer(reference_values=ref_df)
        result = imputer.impute(df)

        assert not result.isna().any().any()
        assert result["a"].iloc[1] == 20.0  # Mean of reference
        assert result["b"].iloc[0] == 200.0

    def test_cold_deck_none_fallback(self):
        """Test cold deck with None (fallback to median)."""
        df = pd.DataFrame({"a": [1, np.nan, 3, 5]})
        imputer = ColdDeckImputer(reference_values=None)
        result = imputer.impute(df)

        assert not result.isna().any().any()
        assert result["a"].iloc[1] == 3.0  # Median

    def test_cold_deck_with_array(self):
        """Test cold deck with array reference values."""
        df = pd.DataFrame({"a": [1, np.nan, 3]})
        imputer = ColdDeckImputer(reference_values={"a": np.array([10, 20, 30])})
        result = imputer.impute(df)

        assert not result.isna().any().any()
        # Should sample from reference array
        assert result["a"].iloc[1] in [10, 20, 30]


class TestHybridImputer:
    """Tests for HybridImputer."""

    def test_hybrid_default(self):
        """Test hybrid imputer with default methods."""
        df = pd.DataFrame({"a": [1, np.nan, 3, np.nan, 5]})
        imputer = HybridImputer()
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_hybrid_custom_chain(self):
        """Test hybrid with custom method chain."""
        df = pd.DataFrame({"a": [1, np.nan, np.nan, 4, np.nan]})
        imputer = HybridImputer(methods=[InterpolationImputer(), MeanImputer()])
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_hybrid_fallback_chain(self):
        """Test that hybrid tries multiple methods."""
        df = pd.DataFrame({"a": [np.nan, np.nan, 3, 4, 5]})
        # Interpolation might struggle with leading NaNs, should fall back
        imputer = HybridImputer(methods=[InterpolationImputer(), MeanImputer()])
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_hybrid_all_nan_fallback(self):
        """Test hybrid final fallback when all values missing."""
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        imputer = HybridImputer()
        result = imputer.impute(df)

        # Should fall back to 0
        assert all(result["a"] == 0)


# ============================================================================
# Integration Tests
# ============================================================================


class TestPhaseMethodsIntegration:
    """Integration tests for all phase methods."""

    def test_all_phase_methods_fill_nans(self):
        """Test that all phase methods successfully fill NaNs."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})

        imputers = [
            ModeImputer(),
            ConstantImputer(fill_value=0),
            EndOfDistributionImputer(),
            WeightedMovingAverageImputer(),
            LinearTrendImputer(),
            PolynomialTrendImputer(),
            KalmanFilterImputer(),
            ColdDeckImputer(),
            HybridImputer(),
        ]

        for imputer in imputers:
            result = imputer.impute(df)
            assert not result.isna().any().any(), (
                f"{imputer.__class__.__name__} failed to fill all NaNs"
            )

    def test_all_methods_preserve_shape(self):
        """Test that all methods preserve DataFrame shape."""
        df = pd.DataFrame(
            {"a": [1, np.nan, 3], "b": [np.nan, 2, 3], "c": [1, 2, np.nan]}
        )

        imputers = [
            ModeImputer(),
            ConstantImputer(),
            EndOfDistributionImputer(),
            WeightedMovingAverageImputer(),
            LinearTrendImputer(),
            PolynomialTrendImputer(),
            KalmanFilterImputer(),
            ColdDeckImputer(),
            HybridImputer(),
        ]

        for imputer in imputers:
            result = imputer.impute(df)
            assert result.shape == df.shape, (
                f"{imputer.__class__.__name__} changed shape"
            )
            assert list(result.columns) == list(df.columns), (
                f"{imputer.__class__.__name__} changed columns"
            )
