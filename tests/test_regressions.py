"""Regression tests for behaviour fixed while preparing the first release."""

import numpy as np
import pandas as pd
import pytest

from imputation_methods import (
    ColdDeckImputer,
    GroupMeanImputer,
    HotDeckImputer,
    PMMImputer,
    SeasonalImputer,
    cold_deck_impute,
)


def test_pmm_imputes_only_observed_values_and_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    df = pd.DataFrame({"x": x, "y": np.round(2 * x + rng.normal(size=60), 1)})
    df.loc[::5, "y"] = np.nan

    first = PMMImputer(k=5, random_state=7).impute(df)
    second = PMMImputer(k=5, random_state=7).impute(df)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["y"]) <= set(df["y"].dropna())


def test_pmm_leaves_fully_missing_column_untouched() -> None:
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1.0, 2.0, 3.0]})
    assert PMMImputer(k=2, random_state=0).impute(df)["a"].isna().all()


def test_cold_deck_array_reference_is_reproducible() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan, np.nan, 5.0]})
    reference = {"a": np.array([10.0, 20.0, 30.0])}

    first = ColdDeckImputer(reference_values=reference, random_state=3).impute(df)
    second = cold_deck_impute(df, reference_values=reference, random_state=3)

    pd.testing.assert_frame_equal(first, second)
    assert set(first.loc[1:3, "a"]) <= {10.0, 20.0, 30.0}


def test_seasonal_imputer_uses_per_phase_statistic() -> None:
    df = pd.DataFrame({"a": [1.0, 10.0, np.nan, 20.0, 3.0, np.nan]})
    result = SeasonalImputer(period=2, method="median").impute(df)
    assert result["a"].tolist() == [1.0, 10.0, 2.0, 20.0, 3.0, 15.0]


def test_group_mean_imputer_with_global_fallback() -> None:
    df = pd.DataFrame(
        {
            "group": [1, 1, 2, 2, 3],
            "value": [10.0, np.nan, 20.0, np.nan, np.nan],
        }
    )
    result = GroupMeanImputer(group_col="group").impute(df)
    assert result["value"].tolist() == [10.0, 10.0, 20.0, 20.0, 15.0]


@pytest.mark.parametrize("stratify_cols", [["g"], ["g", "h"]])
def test_hot_deck_stratified_draws_from_group(stratify_cols: list[str]) -> None:
    df = pd.DataFrame(
        {
            "g": [0, 0, 0, 1, 1, 1],
            "h": [0, 0, 0, 0, 0, 0],
            "v": [1.0, 1.0, np.nan, 9.0, 9.0, np.nan],
        }
    )
    result = HotDeckImputer(stratify_cols=stratify_cols, random_state=0).impute(df)
    assert result["v"].tolist() == [1.0, 1.0, 1.0, 9.0, 9.0, 9.0]
