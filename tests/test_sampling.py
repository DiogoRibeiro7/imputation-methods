"""Tests for the donor-based sampling imputers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imputation_methods import (
    ColdDeckImputer,
    HotDeckImputer,
    RandomSamplingImputer,
    cold_deck_impute,
)


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


class TestHotDeckImputer:
    """Tests for ``HotDeckImputer``."""

    def test_hot_deck_imputer(self) -> None:
        df = pd.DataFrame({"group": [0, 0, 1, 1], "a": [1.0, np.nan, 3.0, np.nan]})
        imputed = HotDeckImputer(
            stratify_cols=["group"],
            random_state=0,
        ).impute(df)
        assert not imputed["a"].isna().any()

    def test_hot_deck_imputer_no_stratification(self) -> None:
        """Test hot deck imputer without stratification columns."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, np.nan, 7, 8]})
        imputed = HotDeckImputer(random_state=0).impute(df)
        assert not imputed.isna().any().any()

    def test_hot_deck_empty_donors_in_group(self) -> None:
        """Test hot deck when a group has no donors."""
        df = pd.DataFrame({"group": [0, 0, 1, 1], "a": [np.nan, np.nan, 3.0, 4.0]})
        imputer = HotDeckImputer(stratify_cols=["group"], random_state=0)
        imputed = imputer.impute(df)
        # Should fall back to full column donors when group has none
        assert not imputed["a"].isna().any()

    @pytest.mark.parametrize("stratify_cols", [["g"], ["g", "h"]])
    def test_hot_deck_stratified_draws_from_group(
        self, stratify_cols: list[str]
    ) -> None:
        df = pd.DataFrame(
            {
                "g": [0, 0, 0, 1, 1, 1],
                "h": [0, 0, 0, 0, 0, 0],
                "v": [1.0, 1.0, np.nan, 9.0, 9.0, np.nan],
            }
        )
        result = HotDeckImputer(stratify_cols=stratify_cols, random_state=0).impute(df)
        assert result["v"].tolist() == [1.0, 1.0, 1.0, 9.0, 9.0, 9.0]


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

    def test_cold_deck_array_reference_is_reproducible(self) -> None:
        df = pd.DataFrame({"a": [1.0, np.nan, np.nan, np.nan, 5.0]})
        reference = {"a": np.array([10.0, 20.0, 30.0])}

        first = ColdDeckImputer(reference_values=reference, random_state=3).impute(df)
        second = cold_deck_impute(df, reference_values=reference, random_state=3)

        pd.testing.assert_frame_equal(first, second)
        assert set(first.loc[1:3, "a"]) <= {10.0, 20.0, 30.0}
