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
    "GAINImputer": {"max_iter": 200, "random_state": 0},
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


@pytest.fixture
def single_column() -> pd.DataFrame:
    """One incomplete column, with gaps inside and at both ends."""
    return pd.DataFrame(
        {"a": [np.nan, 1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, np.nan]},
        index=pd.RangeIndex(10, 20),
    )


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
    [imputer for imputer in IMPUTERS if imputer.__name__ != "GroupMeanImputer"],
    ids=lambda c: c.__name__,
)
def test_handles_a_single_column(
    cls: type[BaseImputer], single_column: pd.DataFrame
) -> None:
    """A lone column must not crash; multivariate methods may leave gaps."""
    kwargs = {
        key: value
        for key, value in KWARGS.get(cls.__name__, {}).items()
        if key != "stratify_cols"
    }
    snapshot = single_column.copy()

    result = cls(**kwargs).impute(single_column)

    pd.testing.assert_frame_equal(single_column, snapshot, obj="input")
    pd.testing.assert_index_equal(result.index, single_column.index)
    assert result.columns[0] == "a"
    observed = snapshot["a"].notna().to_numpy()
    np.testing.assert_allclose(
        result["a"].to_numpy(dtype=float)[observed], snapshot["a"].to_numpy()[observed]
    )


@pytest.mark.parametrize("cls", IMPUTERS, ids=lambda c: c.__name__)
def test_output_dtypes_follow_the_policy(
    cls: type[BaseImputer], frame: pd.DataFrame
) -> None:
    """Complete columns come back unchanged; imputed columns as floating point."""
    mixed = frame.assign(
        a=frame["a"].astype("float32"),
        b=frame["b"].astype("Float64"),
        c=frame["c"].mul(10).round().astype("Int64"),
        complete=pd.array(range(len(frame)), dtype="Int64"),
    )

    result = _build(cls).impute(mixed)

    assert str(result["a"].dtype) == "float32"
    assert str(result["b"].dtype) == "Float64"
    assert str(result["c"].dtype) == "Float64"
    pd.testing.assert_series_equal(result["g"], mixed["g"])
    pd.testing.assert_series_equal(result["complete"], mixed["complete"])


# Imputers that fill with a constant chosen by the user, so they need no data.
FILLS_EMPTY_COLUMNS = {"ConstantImputer"}


@pytest.mark.parametrize(
    "cls",
    [cls for cls in IMPUTERS if cls.__name__ not in FILLS_EMPTY_COLUMNS],
    ids=lambda c: c.__name__,
)
def test_column_without_observed_values_stays_missing(
    cls: type[BaseImputer], frame: pd.DataFrame
) -> None:
    result = _build(cls).impute(frame.assign(empty=np.nan))

    assert result["empty"].isna().all()


@pytest.mark.parametrize(
    "cls",
    [
        imputation_methods.BaggingImputer,
        imputation_methods.EMImputer,
        imputation_methods.GAINImputer,
        imputation_methods.GaussianProcessImputer,
        imputation_methods.KNNImputer,
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
