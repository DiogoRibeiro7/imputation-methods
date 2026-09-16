"""Tests for the order-dependent time-series imputers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imputation_methods import (
    ForwardFillFallbackImputer,
    InterpolationImputer,
    KalmanFilterImputer,
    LinearTrendImputer,
    LOCFImputer,
    MovingAverageImputer,
    NOCBImputer,
    PolynomialTrendImputer,
    SeasonalImputer,
    WeightedMovingAverageImputer,
)


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


class TestMovingAverageImputer:
    """Tests for MovingAverageImputer."""

    def test_mean_window(self):
        """Test moving average with mean."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0]})
        imputer = MovingAverageImputer(window=3, strategy="mean")
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_median_window(self):
        """Test moving average with median."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0]})
        imputer = MovingAverageImputer(window=3, strategy="median")
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_window_size_validation(self):
        """Test window size validation."""
        with pytest.raises(ValueError, match="window must be >= 1"):
            MovingAverageImputer(window=0)

    def test_invalid_method(self):
        """Test invalid method."""
        with pytest.raises(ValueError, match="strategy must be 'mean' or 'median'"):
            MovingAverageImputer(strategy="invalid")

    def test_centered_window(self):
        """Test centered window."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0]})
        imputer = MovingAverageImputer(window=3, center=True)
        result = imputer.impute(df)

        assert not result.isna().any().any()


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


class TestSeasonalImputer:
    """Tests for SeasonalImputer."""

    def test_weekly_seasonality(self):
        """Test weekly seasonal pattern."""
        # Create data with weekly pattern
        data = [100, 120, 110, np.nan, 130, 90, 80] * 2
        df = pd.DataFrame({"sales": data})
        imputer = SeasonalImputer(period=7, strategy="median")
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_daily_seasonality(self):
        """Test daily seasonal pattern (hourly data)."""
        # 24 hours, repeated
        data = list(range(24)) * 2
        data[5] = np.nan
        data[29] = np.nan
        df = pd.DataFrame({"temp": data})

        imputer = SeasonalImputer(period=24, strategy="mean")
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_period_validation(self):
        """Test period validation."""
        with pytest.raises(ValueError, match="period must be >= 2"):
            SeasonalImputer(period=1)

    def test_invalid_method(self):
        """Test invalid aggregation method."""
        with pytest.raises(ValueError, match="strategy must be 'mean' or 'median'"):
            SeasonalImputer(strategy="invalid")

    def test_mean_vs_median(self):
        """Test mean vs median methods."""
        data = [1, 2, np.nan, 4] * 2
        df = pd.DataFrame({"a": data})

        imputer_mean = SeasonalImputer(period=4, strategy="mean")
        result_mean = imputer_mean.impute(df)

        imputer_median = SeasonalImputer(period=4, strategy="median")
        result_median = imputer_median.impute(df)

        assert not result_mean.isna().any().any()
        assert not result_median.isna().any().any()

    def test_seasonal_imputer_uses_per_phase_statistic(self) -> None:
        df = pd.DataFrame({"a": [1.0, 10.0, np.nan, 20.0, 3.0, np.nan]})
        result = SeasonalImputer(period=2, strategy="median").impute(df)
        assert result["a"].tolist() == [1.0, 10.0, 2.0, 20.0, 3.0, 15.0]


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


class TestLOCFAndNOCBImputers:
    """Tests for ``LOCFImputer`` and ``NOCBImputer``."""

    def test_locf_nocb(self) -> None:
        df = pd.DataFrame({"a": [np.nan, 1, np.nan, 3, np.nan]})
        locf = LOCFImputer().impute(df)
        assert pd.isna(locf.loc[0, "a"])
        assert locf.loc[2, "a"] == 1
        nocb = NOCBImputer().impute(df)
        assert nocb.loc[0, "a"] == 1
        assert nocb.loc[2, "a"] == 3

    def test_locf_all_nan(self) -> None:
        """Test LOCF with all NaN values."""
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        imputed = LOCFImputer().impute(df)
        assert imputed.isna().all().all()

    def test_nocb_all_nan(self) -> None:
        """Test NOCB with all NaN values."""
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        imputed = NOCBImputer().impute(df)
        assert imputed.isna().all().all()

    def test_locf_trailing_nan(self) -> None:
        """Test LOCF with trailing NaN values."""
        df = pd.DataFrame({"a": [1, 2, np.nan, np.nan]})
        imputed = LOCFImputer().impute(df)
        assert imputed.loc[2, "a"] == 2
        assert imputed.loc[3, "a"] == 2

    def test_nocb_leading_nan(self) -> None:
        """Test NOCB with leading NaN values."""
        df = pd.DataFrame({"a": [np.nan, np.nan, 3, 4]})
        imputed = NOCBImputer().impute(df)
        assert imputed.loc[0, "a"] == 3
        assert imputed.loc[1, "a"] == 3
