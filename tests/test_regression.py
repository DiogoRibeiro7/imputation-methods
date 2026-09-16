"""Tests for the regression-based imputers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imputation_methods import (
    BayesianRidgeImputer,
    GaussianProcessImputer,
    HuberImputer,
    PMMImputer,
    RANSACImputer,
    RegressionImputer,
    StochasticRegressionImputer,
)


class TestRegressionImputer:
    """Tests for ``RegressionImputer``."""

    def test_regression_imputer(self) -> None:
        df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, 6, 7, 8]})
        imputed = RegressionImputer().impute(df)
        assert not imputed.isna().any().any()

    def test_regression_imputer_single_column(self) -> None:
        """Test regression imputer with single column (no predictors)."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4]})
        imputed = RegressionImputer().impute(df)
        # Should leave NaN as is when no predictors available
        assert imputed.isna().any().any()


class TestStochasticRegressionImputer:
    """Tests for ``StochasticRegressionImputer``."""

    def test_stochastic_regression_imputer(self) -> None:
        df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, 6, 7, 8]})
        imputed = StochasticRegressionImputer(random_state=0).impute(df)
        assert not imputed.isna().any().any()

    def test_stochastic_regression_single_column(self) -> None:
        """Test stochastic regression imputer with single column."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4]})
        imputed = StochasticRegressionImputer(random_state=0).impute(df)
        # Should leave NaN as is when no predictors available
        assert imputed.isna().any().any()


class TestPMMImputer:
    """Tests for ``PMMImputer``."""

    def test_pmm_impute_basic(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3, 4, np.nan], "y": [5, 6, 7, 8, 9]})
        imputed = PMMImputer(n_neighbors=2, random_state=0).impute(df)
        assert not imputed.isna().any().any()

    def test_pmm_imputer_no_predictors(self) -> None:
        """Test PMM imputer when no predictors are available."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4]})
        imputed = PMMImputer(n_neighbors=2, random_state=0).impute(df)
        # Should fall back to mean imputation
        assert not imputed.isna().any().any()
        assert np.isclose(imputed.loc[2, "a"], (1 + 2 + 4) / 3)

    def test_pmm_imputer_empty_observed(self) -> None:
        """Test PMM when observed data is empty for predictors."""
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1, 2, 3]})
        imputed = PMMImputer(n_neighbors=2, random_state=0).impute(df)
        # Should handle gracefully
        assert imputed.isna().all()["a"]

    def test_pmm_invalid_k_type(self) -> None:
        """Test PMM imputer rejects non-integer k values."""
        with pytest.raises(TypeError, match="n_neighbors must be an integer"):
            PMMImputer(n_neighbors=3.5)  # type: ignore

    def test_pmm_invalid_k_value(self) -> None:
        """Test PMM imputer rejects non-positive k values."""
        with pytest.raises(ValueError, match="n_neighbors must be positive"):
            PMMImputer(n_neighbors=0)
        with pytest.raises(ValueError, match="n_neighbors must be positive"):
            PMMImputer(n_neighbors=-2)

    def test_pmm_imputes_only_observed_values_and_is_reproducible(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=60)
        df = pd.DataFrame({"x": x, "y": np.round(2 * x + rng.normal(size=60), 1)})
        df.loc[::5, "y"] = np.nan

        first = PMMImputer(n_neighbors=5, random_state=7).impute(df)
        second = PMMImputer(n_neighbors=5, random_state=7).impute(df)

        pd.testing.assert_frame_equal(first, second)
        assert set(first["y"]) <= set(df["y"].dropna())

    def test_pmm_leaves_fully_missing_column_untouched(self) -> None:
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1.0, 2.0, 3.0]})
        assert PMMImputer(n_neighbors=2, random_state=0).impute(df)["a"].isna().all()


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


class TestGaussianProcessImputer:
    """Tests for ``GaussianProcessImputer``."""

    def test_gaussian_process_imputer(self) -> None:
        df = pd.DataFrame({"a": [1, 2, np.nan], "b": [4, 5, 6]})
        imputed = GaussianProcessImputer(random_state=0).impute(df)
        assert not imputed.isna().any().any()

    def test_gaussian_process_single_column(self) -> None:
        """Test Gaussian process imputer with single column."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4]})
        imputed = GaussianProcessImputer(random_state=0).impute(df)
        # Should leave NaN as is when no predictors available
        assert imputed.isna().any().any()
