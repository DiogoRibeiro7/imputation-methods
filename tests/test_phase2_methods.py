"""Tests for Phase 2 imputation methods (8 methods total)."""

import numpy as np
import pandas as pd
import pytest

from imputation_methods import (
    BaggingImputer,
    # Phase 2.1: Bayesian & Ensemble
    BayesianRidgeImputer,
    # Phase 2.3: Robust & Outlier-Aware
    HuberImputer,
    LocalMeanImputer,
    # For ensemble tests
    MeanImputer,
    MedianImputer,
    # Phase 2.2: Spatial & Distance
    RadiusNeighborsImputer,
    RANSACImputer,
    StackingImputer,
    TrimmedMeanImputer,
)

# ============================================================================
# PHASE 2.1: Bayesian & Ensemble Methods
# ============================================================================


class TestBayesianRidgeImputer:
    """Tests for BayesianRidgeImputer."""

    def test_basic_imputation(self):
        """Test basic Bayesian ridge imputation."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5], "b": [2, 4, 6, np.nan, 10]})
        imputer = BayesianRidgeImputer()
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_single_column(self):
        """Test with single column falls back gracefully."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})
        imputer = BayesianRidgeImputer()
        result = imputer.impute(df)

        assert not result.isna().any().any()


class TestStackingImputer:
    """Tests for StackingImputer."""

    def test_default_stacking(self):
        """Test stacking with default imputers."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})
        imputer = StackingImputer()
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_custom_base_imputers(self):
        """Test with custom base imputers."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})
        imputer = StackingImputer(base_imputers=[MeanImputer(), MedianImputer()])
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_meta_strategy_median(self):
        """Test median meta strategy."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})
        imputer = StackingImputer(meta_strategy="median")
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_invalid_meta_strategy(self):
        """Test invalid meta strategy raises error."""
        with pytest.raises(ValueError, match="meta_strategy must be"):
            StackingImputer(meta_strategy="invalid")


class TestBaggingImputer:
    """Tests for BaggingImputer."""

    def test_basic_bagging(self):
        """Test basic bagging imputation."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5, np.nan, 7]})
        imputer = BaggingImputer(n_estimators=5, random_state=42)
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_with_custom_base_imputer(self):
        """Test bagging with custom base imputer."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})
        imputer = BaggingImputer(base_imputer=MedianImputer(), n_estimators=3)
        result = imputer.impute(df)

        assert not result.isna().any().any()


# ============================================================================
# PHASE 2.2: Spatial & Distance-Based Methods
# ============================================================================


class TestRadiusNeighborsImputer:
    """Tests for RadiusNeighborsImputer."""

    def test_basic_radius_imputation(self):
        """Test basic radius neighbors imputation."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5], "b": [2, 4, 6, np.nan, 10]})
        imputer = RadiusNeighborsImputer(radius=5.0)
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_large_radius(self):
        """Test with large radius includes all neighbors."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})
        imputer = RadiusNeighborsImputer(radius=100.0)
        result = imputer.impute(df)

        assert not result.isna().any().any()


class TestLocalMeanImputer:
    """Tests for LocalMeanImputer."""

    def test_basic_local_mean(self):
        """Test basic local weighted mean."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})
        imputer = LocalMeanImputer(n_neighbors=3)
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_distance_weight_power(self):
        """Test different distance weighting powers."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})
        imputer = LocalMeanImputer(distance_weight_power=1.0)
        result = imputer.impute(df)

        assert not result.isna().any().any()


# ============================================================================
# PHASE 2.3: Robust & Outlier-Aware Methods
# ============================================================================


class TestHuberImputer:
    """Tests for HuberImputer."""

    def test_basic_huber(self):
        """Test basic Huber regression imputation."""
        df = pd.DataFrame(
            {
                "a": [1, 2, np.nan, 100, 5],  # 100 is outlier
                "b": [2, 4, 6, 200, np.nan],
            }
        )
        imputer = HuberImputer()
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_custom_epsilon(self):
        """Test with custom epsilon parameter."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})
        imputer = HuberImputer(epsilon=2.0)
        result = imputer.impute(df)

        assert not result.isna().any().any()


class TestRANSACImputer:
    """Tests for RANSACImputer."""

    def test_basic_ransac(self):
        """Test basic RANSAC imputation."""
        df = pd.DataFrame(
            {
                "a": [1, 2, np.nan, 100, 5, 6],  # 100 is outlier
                "b": [2, 4, 6, 200, 10, np.nan],
            }
        )
        imputer = RANSACImputer(random_state=42)
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_max_trials(self):
        """Test with custom max_trials."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5, 6]})
        imputer = RANSACImputer(max_trials=50)
        result = imputer.impute(df)

        assert not result.isna().any().any()


class TestTrimmedMeanImputer:
    """Tests for TrimmedMeanImputer."""

    def test_basic_trimmed_mean(self):
        """Test basic trimmed mean imputation."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 100]})  # 100 is outlier
        imputer = TrimmedMeanImputer(trim_fraction=0.2)
        result = imputer.impute(df)

        assert not result.isna().any().any()
        # Verify outlier was excluded from calculation
        assert result["a"].iloc[2] < 50  # Much less than 100

    def test_no_trimming(self):
        """Test with no trimming (trim_fraction=0)."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})
        imputer = TrimmedMeanImputer(trim_fraction=0.0)
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_invalid_trim_fraction(self):
        """Test invalid trim fraction raises error."""
        with pytest.raises(ValueError, match="trim_fraction must be"):
            TrimmedMeanImputer(trim_fraction=0.6)


# ============================================================================
# Integration Tests
# ============================================================================


class TestPhase2Integration:
    """Integration tests for all Phase 2 methods."""

    def test_all_methods_fill_nans(self):
        """Test that all Phase 2 methods fill NaNs."""
        df = pd.DataFrame(
            {"a": [1, 2, np.nan, 4, 5, 6], "b": [2, 4, 6, np.nan, 10, 12]}
        )

        imputers = [
            BayesianRidgeImputer(),
            StackingImputer(),
            BaggingImputer(n_estimators=3),
            RadiusNeighborsImputer(radius=10.0),
            LocalMeanImputer(),
            HuberImputer(),
            RANSACImputer(),
            TrimmedMeanImputer(),
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
            BayesianRidgeImputer(),
            StackingImputer(),
            BaggingImputer(),
            RadiusNeighborsImputer(radius=10.0),
            LocalMeanImputer(),
            HuberImputer(),
            RANSACImputer(),
            TrimmedMeanImputer(),
        ]

        for imputer in imputers:
            result = imputer.impute(df)
            assert result.shape == df.shape, (
                f"{imputer.__class__.__name__} changed shape"
            )
            assert list(result.columns) == list(df.columns), (
                f"{imputer.__class__.__name__} changed columns"
            )
