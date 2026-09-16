"""Behaviour every imputer must share, checked across the whole public API."""

import inspect
import logging

import numpy as np
import pandas as pd
import pytest

import imputation_methods
from imputation_methods import BaseImputer

# Constructor arguments for imputers that cannot be built with defaults, or
# whose defaults are too slow for a contract test.
KWARGS: dict[str, dict[str, object]] = {
    "GroupMeanImputer": {"group_col": "g"},
    "HotDeckImputer": {"stratify_cols": ["g"], "random_state": 0},
    "AutoencoderImputer": {"max_iter": 50, "random_state": 0},
    "BaggingImputer": {"n_estimators": 2, "random_state": 0},
    "GAINImputer": {"iterations": 200, "random_state": 0},
    "MissForestImputer": {"random_state": 0},
    "RadiusNeighborsImputer": {"radius": 10.0},
}

IMPUTERS = sorted(
    (
        obj
        for obj in vars(imputation_methods).values()
        if inspect.isclass(obj)
        and issubclass(obj, BaseImputer)
        and obj is not BaseImputer
    ),
    key=lambda cls: cls.__name__,
)


def _build(cls: type[BaseImputer]) -> BaseImputer:
    return cls(**KWARGS.get(cls.__name__, {}))


@pytest.fixture
def frame() -> pd.DataFrame:
    """Several incomplete columns; first and last rows complete for LOCF/NOCB."""
    rng = np.random.default_rng(42)
    n = 40
    base = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "g": np.repeat([0, 1], n // 2),
            "a": base + rng.normal(scale=0.1, size=n),
            "b": 2 * base + rng.normal(scale=0.1, size=n),
            "c": -base + rng.normal(scale=0.1, size=n),
        },
        index=pd.RangeIndex(100, 100 + n),
    )
    for column in ["a", "b", "c"]:
        rows = rng.choice(np.arange(1, n - 1), size=6, replace=False)
        df.loc[df.index[rows], column] = np.nan
    return df


@pytest.fixture(autouse=True)
def _quiet_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger="imputation_methods")


@pytest.mark.parametrize("cls", IMPUTERS, ids=lambda c: c.__name__)
def test_fills_several_incomplete_columns(
    cls: type[BaseImputer], frame: pd.DataFrame
) -> None:
    snapshot = frame.copy()

    result = _build(cls).impute(frame)

    pd.testing.assert_frame_equal(frame, snapshot, obj="input")
    assert result is not frame
    pd.testing.assert_index_equal(result.index, frame.index)
    assert list(result.columns[: frame.shape[1]]) == list(frame.columns)
    assert result[frame.columns].notna().all().all()
    observed = snapshot.notna().to_numpy()
    np.testing.assert_allclose(
        result[frame.columns].to_numpy(dtype=float)[observed],
        snapshot.to_numpy(dtype=float)[observed],
    )


@pytest.mark.parametrize(
    "cls",
    [
        imputation_methods.BaggingImputer,
        imputation_methods.EMImputer,
        imputation_methods.GAINImputer,
        imputation_methods.GaussianProcessImputer,
        imputation_methods.KNNImputerMethod,
        imputation_methods.MICEImputer,
        imputation_methods.MissForestImputer,
        imputation_methods.PMMImputer,
        imputation_methods.RegressionImputer,
    ],
    ids=lambda c: c.__name__,
)
def test_empty_column_does_not_change_other_columns(
    cls: type[BaseImputer], frame: pd.DataFrame
) -> None:
    with_empty = frame.assign(empty=np.nan)
    kwargs = {
        **(
            {"random_state": 0}
            if "random_state" in inspect.signature(cls).parameters
            else {}
        ),
        **KWARGS.get(cls.__name__, {}),
    }

    expected = cls(**kwargs).impute(frame)
    result = cls(**kwargs).impute(with_empty)

    assert result["empty"].isna().all()
    pd.testing.assert_frame_equal(
        result[frame.columns], expected[frame.columns], check_dtype=False
    )
