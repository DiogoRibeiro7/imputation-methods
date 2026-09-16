"""Tests for new imputation methods added to the library."""

import numpy as np
import pandas as pd
import pytest

from imputation_methods import (
    EMImputer,
    ForwardFillFallbackImputer,
    IndicatorImputer,
    InterpolationImputer,
    MovingAverageImputer,
    QuantileImputer,
    RandomSamplingImputer,
    SeasonalImputer,
)


class TestInterpolationImputer:
    """Tests for InterpolationImputer."""

    def test_linear_interpolation(self):
        """Test linear interpolation."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, np.nan, 5.0]})
        imputer = InterpolationImputer(method="linear")
        result = imputer.impute(df)

        assert not result.isna().any().any()
        assert result["a"].iloc[2] == pytest.approx(3.0)
        assert result["a"].iloc[3] == pytest.approx(4.0)

    def test_polynomial_interpolation(self):
        """Test polynomial interpolation."""
        df = pd.DataFrame({"a": [1.0, 4.0, np.nan, 16.0, 25.0]})
        imputer = InterpolationImputer(method="polynomial", order=2)
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_spline_interpolation(self):
        """Test spline interpolation."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0]})
        imputer = InterpolationImputer(method="spline", order=2)
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_limit_parameter(self):
        """Test limit parameter."""
        df = pd.DataFrame({"a": [1.0, np.nan, np.nan, np.nan, 5.0]})
        imputer = InterpolationImputer(method="linear", limit=1)
        result = imputer.impute(df)

        # Should fill limited consecutive NaNs, then use fallback
        assert not result.isna().any().any()

    def test_invalid_method(self):
        """Test invalid interpolation method."""
        with pytest.raises(ValueError, match="method must be one of"):
            InterpolationImputer(method="invalid")

    def test_preserves_observed(self):
        """Test that observed values are preserved."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        imputer = InterpolationImputer()
        result = imputer.impute(df)

        observed_mask = ~df["a"].isna()
        pd.testing.assert_series_equal(
            df.loc[observed_mask, "a"], result.loc[observed_mask, "a"]
        )


class TestEMImputer:
    """Tests for EMImputer."""

    def test_basic_imputation(self):
        """Test basic EM imputation."""
        df = pd.DataFrame(
            {"a": [1.0, 2.0, np.nan, 4.0, 5.0], "b": [5.0, np.nan, 7.0, 8.0, 9.0]}
        )
        imputer = EMImputer(max_iter=50)
        result = imputer.impute(df)

        assert not result.isna().any().any()
        assert result.shape == df.shape

    def test_convergence_parameters(self):
        """Test that convergence parameters are respected."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [5.0, np.nan, 7.0, 8.0]})
        imputer = EMImputer(max_iter=10, tol=1e-3)
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_reproducibility(self):
        """Test that results are reproducible with random_state."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})

        imputer1 = EMImputer(random_state=42)
        result1 = imputer1.impute(df)

        imputer2 = EMImputer(random_state=42)
        result2 = imputer2.impute(df)

        pd.testing.assert_frame_equal(result1, result2)


class TestMovingAverageImputer:
    """Tests for MovingAverageImputer."""

    def test_mean_window(self):
        """Test moving average with mean."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0]})
        imputer = MovingAverageImputer(window=3, method="mean")
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_median_window(self):
        """Test moving average with median."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0]})
        imputer = MovingAverageImputer(window=3, method="median")
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_window_size_validation(self):
        """Test window size validation."""
        with pytest.raises(ValueError, match="window must be >= 1"):
            MovingAverageImputer(window=0)

    def test_invalid_method(self):
        """Test invalid method."""
        with pytest.raises(ValueError, match="method must be 'mean' or 'median'"):
            MovingAverageImputer(method="invalid")

    def test_centered_window(self):
        """Test centered window."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0]})
        imputer = MovingAverageImputer(window=3, center=True)
        result = imputer.impute(df)

        assert not result.isna().any().any()


class TestRandomSamplingImputer:
    """Tests for RandomSamplingImputer."""

    def test_basic_sampling(self):
        """Test basic random sampling."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, np.nan, 6.0]})
        imputer = RandomSamplingImputer(random_state=42)
        result = imputer.impute(df)

        assert not result.isna().any().any()
        # Imputed values should be from observed values
        assert result["a"].iloc[2] in [1.0, 2.0, 4.0, 6.0]
        assert result["a"].iloc[4] in [1.0, 2.0, 4.0, 6.0]

    def test_reproducibility(self):
        """Test that results are reproducible."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})

        imputer1 = RandomSamplingImputer(random_state=42)
        result1 = imputer1.impute(df)

        imputer2 = RandomSamplingImputer(random_state=42)
        result2 = imputer2.impute(df)

        pd.testing.assert_frame_equal(result1, result2)

    def test_preserves_distribution(self):
        """Test that sampling preserves value range."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan, np.nan]})
        imputer = RandomSamplingImputer(random_state=42)
        result = imputer.impute(df)

        # All imputed values should be from observed values
        assert result["a"].min() >= 1.0
        assert result["a"].max() <= 3.0


class TestIndicatorImputer:
    """Tests for IndicatorImputer."""

    def test_adds_indicator_columns(self):
        """Test that indicator columns are added."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [5.0, np.nan, 7.0, 8.0]})
        imputer = IndicatorImputer(strategy="mean")
        result = imputer.impute(df)

        # Should have original columns + indicator columns
        assert "missing_a" in result.columns
        assert "missing_b" in result.columns
        assert len(result.columns) == 4

    def test_indicator_values(self):
        """Test that indicator values are correct."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        imputer = IndicatorImputer()
        result = imputer.impute(df)

        assert result["missing_a"].iloc[0] == 0
        assert result["missing_a"].iloc[1] == 1
        assert result["missing_a"].iloc[2] == 0

    def test_mean_strategy(self):
        """Test mean imputation strategy."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        imputer = IndicatorImputer(strategy="mean")
        result = imputer.impute(df)

        expected_mean = (1.0 + 2.0 + 4.0) / 3
        assert result["a"].iloc[2] == pytest.approx(expected_mean)

    def test_median_strategy(self):
        """Test median imputation strategy."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        imputer = IndicatorImputer(strategy="median")
        result = imputer.impute(df)

        assert result["a"].iloc[2] == 2.0

    def test_zero_strategy(self):
        """Test zero imputation strategy."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        imputer = IndicatorImputer(strategy="zero")
        result = imputer.impute(df)

        assert result["a"].iloc[1] == 0.0

    def test_custom_prefix(self):
        """Test custom indicator prefix."""
        df = pd.DataFrame({"a": [1.0, np.nan]})
        imputer = IndicatorImputer(indicator_prefix="was_missing_")
        result = imputer.impute(df)

        assert "was_missing_a" in result.columns

    def test_invalid_strategy(self):
        """Test invalid strategy."""
        with pytest.raises(ValueError, match="strategy must be"):
            IndicatorImputer(strategy="invalid")


class TestSeasonalImputer:
    """Tests for SeasonalImputer."""

    def test_weekly_seasonality(self):
        """Test weekly seasonal pattern."""
        # Create data with weekly pattern
        data = [100, 120, 110, np.nan, 130, 90, 80] * 2
        df = pd.DataFrame({"sales": data})
        imputer = SeasonalImputer(period=7, method="median")
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_daily_seasonality(self):
        """Test daily seasonal pattern (hourly data)."""
        # 24 hours, repeated
        data = list(range(24)) * 2
        data[5] = np.nan
        data[29] = np.nan
        df = pd.DataFrame({"temp": data})

        imputer = SeasonalImputer(period=24, method="mean")
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_period_validation(self):
        """Test period validation."""
        with pytest.raises(ValueError, match="period must be >= 2"):
            SeasonalImputer(period=1)

    def test_invalid_method(self):
        """Test invalid aggregation method."""
        with pytest.raises(ValueError, match="method must be 'mean' or 'median'"):
            SeasonalImputer(method="invalid")

    def test_mean_vs_median(self):
        """Test mean vs median methods."""
        data = [1, 2, np.nan, 4] * 2
        df = pd.DataFrame({"a": data})

        imputer_mean = SeasonalImputer(period=4, method="mean")
        result_mean = imputer_mean.impute(df)

        imputer_median = SeasonalImputer(period=4, method="median")
        result_median = imputer_median.impute(df)

        assert not result_mean.isna().any().any()
        assert not result_median.isna().any().any()


class TestQuantileImputer:
    """Tests for QuantileImputer."""

    def test_median_quantile(self):
        """Test median (50th percentile) imputation."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0]})
        imputer = QuantileImputer(quantile=0.5)
        result = imputer.impute(df)

        assert not result.isna().any().any()
        assert result["a"].iloc[2] == 3.0  # median of [1,2,4,5]

    def test_quartile_imputation(self):
        """Test 75th percentile imputation."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0]})
        imputer = QuantileImputer(quantile=0.75)
        result = imputer.impute(df)

        assert not result.isna().any().any()
        assert result["a"].iloc[2] == pytest.approx(4.25)

    def test_quantile_validation(self):
        """Test quantile value validation."""
        with pytest.raises(ValueError, match="quantile must be between 0 and 1"):
            QuantileImputer(quantile=1.5)

        with pytest.raises(ValueError, match="quantile must be between 0 and 1"):
            QuantileImputer(quantile=-0.1)

    def test_min_quantile(self):
        """Test minimum (0th percentile)."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        imputer = QuantileImputer(quantile=0.0)
        result = imputer.impute(df)

        assert result["a"].iloc[2] == 1.0

    def test_max_quantile(self):
        """Test maximum (100th percentile)."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        imputer = QuantileImputer(quantile=1.0)
        result = imputer.impute(df)

        assert result["a"].iloc[2] == 4.0


class TestForwardFillFallbackImputer:
    """Tests for ForwardFillFallbackImputer."""

    def test_forward_fill(self):
        """Test forward fill for non-leading NaNs."""
        df = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0]})
        imputer = ForwardFillFallbackImputer(fallback="mean")
        result = imputer.impute(df)

        assert result["a"].iloc[1] == 1.0
        assert result["a"].iloc[2] == 1.0

    def test_fallback_for_leading_nans(self):
        """Test fallback for leading NaNs."""
        df = pd.DataFrame({"a": [np.nan, np.nan, 3.0, 4.0]})
        imputer = ForwardFillFallbackImputer(fallback="mean")
        result = imputer.impute(df)

        assert not result.isna().any().any()
        # First two should be filled with mean of 3 and 4
        expected_mean = 3.5
        assert result["a"].iloc[0] == expected_mean
        assert result["a"].iloc[1] == expected_mean

    def test_mean_fallback(self):
        """Test mean fallback strategy."""
        df = pd.DataFrame({"a": [np.nan, 2.0, 3.0, 4.0]})
        imputer = ForwardFillFallbackImputer(fallback="mean")
        result = imputer.impute(df)

        expected_mean = (2.0 + 3.0 + 4.0) / 3
        assert result["a"].iloc[0] == expected_mean

    def test_median_fallback(self):
        """Test median fallback strategy."""
        df = pd.DataFrame({"a": [np.nan, 2.0, 3.0, 4.0]})
        imputer = ForwardFillFallbackImputer(fallback="median")
        result = imputer.impute(df)

        assert result["a"].iloc[0] == 3.0

    def test_invalid_fallback(self):
        """Test invalid fallback strategy."""
        with pytest.raises(ValueError, match="fallback must be 'mean' or 'median'"):
            ForwardFillFallbackImputer(fallback="invalid")

    def test_no_missing_values(self):
        """Test with no missing values."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        imputer = ForwardFillFallbackImputer()
        result = imputer.impute(df)

        pd.testing.assert_frame_equal(result, df)


class TestNewMethodsIntegration:
    """Integration tests for new methods."""

    def test_all_methods_fill_nans(self):
        """Test that all new methods fill all NaNs."""
        df = pd.DataFrame(
            {"a": [1.0, 2.0, np.nan, 4.0, 5.0], "b": [5.0, np.nan, 7.0, 8.0, 9.0]}
        )

        imputers = [
            InterpolationImputer(),
            EMImputer(max_iter=10),
            MovingAverageImputer(),
            RandomSamplingImputer(random_state=42),
            SeasonalImputer(period=3),
            QuantileImputer(),
            ForwardFillFallbackImputer(),
        ]

        for imputer in imputers:
            result = imputer.impute(df.copy())
            assert not result.isna().any().any(), (
                f"{imputer.__class__.__name__} did not fill all NaNs"
            )

    def test_all_methods_preserve_shape(self):
        """Test that all methods preserve DataFrame shape."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})

        imputers = [
            InterpolationImputer(),
            EMImputer(max_iter=10),
            MovingAverageImputer(),
            RandomSamplingImputer(random_state=42),
            SeasonalImputer(period=2),
            QuantileImputer(),
            ForwardFillFallbackImputer(),
        ]

        for imputer in imputers:
            result = imputer.impute(df.copy())
            # IndicatorImputer adds columns, so exclude it
            if not isinstance(imputer, IndicatorImputer):
                assert result.shape == df.shape, (
                    f"{imputer.__class__.__name__} changed shape"
                )

    def test_methods_with_single_column(self):
        """Test methods with single column DataFrame."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]})

        imputers = [
            InterpolationImputer(),
            MovingAverageImputer(),
            RandomSamplingImputer(random_state=42),
            QuantileImputer(),
            ForwardFillFallbackImputer(),
        ]

        for imputer in imputers:
            result = imputer.impute(df.copy())
            assert not result.isna().any().any(), (
                f"{imputer.__class__.__name__} failed on single column"
            )
