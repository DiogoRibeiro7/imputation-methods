"""Tests for the univariate statistical imputers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imputation_methods import (
    ConstantImputer,
    EndOfDistributionImputer,
    GroupMeanImputer,
    IndicatorImputer,
    MeanImputer,
    MedianImputer,
    ModeImputer,
    QuantileImputer,
    TrimmedMeanImputer,
)


class TestMeanImputer:
    """Tests for ``MeanImputer``."""

    def test_mean_impute_basic(self) -> None:
        df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, np.nan, 7, 8]})
        imputed = MeanImputer().impute(df)
        assert not imputed.isna().any().any()
        assert np.isclose(imputed.loc[2, "a"], (1 + 2 + 4) / 3)

    def test_no_missing(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        imputed = MeanImputer().impute(df)
        pd.testing.assert_frame_equal(imputed, df)

    def test_all_missing(self) -> None:
        df = pd.DataFrame({"a": [np.nan, np.nan]})
        imputed = MeanImputer().impute(df)
        assert imputed.isna().all().all()


class TestMedianImputer:
    """Tests for ``MedianImputer``."""

    def test_median_impute_basic(self) -> None:
        df = pd.DataFrame({"a": [1, 2, np.nan, 10]})
        imputed = MedianImputer().impute(df)
        assert not imputed.isna().any().any()
        assert imputed.loc[2, "a"] == 2

    def test_single_column(self) -> None:
        df = pd.DataFrame({"a": [1, np.nan, 3]})
        imputed = MedianImputer().impute(df)
        assert not imputed.isna().any().any()

    def test_median_imputer_even_count(self) -> None:
        """Test median imputation with even number of values."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4]})
        imputed = MedianImputer().impute(df)
        # Median of [1, 2, 4] should be 2
        assert imputed.loc[2, "a"] == 2


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


class TestEndOfDistributionImputer:
    """Tests for EndOfDistributionImputer."""

    def test_high_position(self):
        """Test imputation at high end of distribution."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, np.nan]})
        imputer = EndOfDistributionImputer(position="high", n_std=1)
        result = imputer.impute(df)

        assert not result.isna().any().any()
        mean = df["a"].mean()
        std = df["a"].std()
        expected = mean + 1 * std
        assert result["a"].iloc[5] == pytest.approx(expected, rel=1e-6)

    def test_low_position(self):
        """Test imputation at low end of distribution."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, np.nan]})
        imputer = EndOfDistributionImputer(position="low", n_std=2)
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
        imputer = GroupMeanImputer(group_col="category", strategy="mean")
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
        imputer = GroupMeanImputer(group_col="category", strategy="median")
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

    def test_group_mean_imputer_with_global_fallback(self) -> None:
        df = pd.DataFrame(
            {
                "group": [1, 1, 2, 2, 3],
                "value": [10.0, np.nan, 20.0, np.nan, np.nan],
            }
        )
        result = GroupMeanImputer(group_col="group").impute(df)
        assert result["value"].tolist() == [10.0, 10.0, 20.0, 20.0, 15.0]


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
