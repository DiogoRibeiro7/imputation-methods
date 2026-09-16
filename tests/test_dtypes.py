"""Output dtype policy: unchanged complete columns, floating-point imputed columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imputation_methods import (
    LOCFImputer,
    MeanImputer,
    MICEImputer,
    RegressionImputer,
    mean_impute,
)


@pytest.fixture
def mixed() -> pd.DataFrame:
    """Incomplete and complete columns in the dtypes users commonly have."""
    rng = np.random.default_rng(0)
    n = 30
    holes = np.zeros(n, dtype=bool)
    holes[[3, 9, 17]] = True
    base = rng.normal(size=n)
    return pd.DataFrame(
        {
            "float64_missing": np.where(holes, np.nan, base + 5),
            "float32_missing": pd.Series(
                np.where(holes, np.nan, 2 * base), dtype="float32"
            ),
            "int64_complete": np.arange(n, dtype="int64") % 5,
            "Int64_missing": pd.array(
                [
                    None if h else int(v)
                    for h, v in zip(holes, np.round(10 * base), strict=True)
                ],
                dtype="Int64",
            ),
            "Float64_missing": pd.array(
                [None if h else float(v) for h, v in zip(holes, base, strict=True)],
                dtype="Float64",
            ),
            "Int64_complete": pd.array(np.arange(n), dtype="Int64"),
        }
    )


EXPECTED = {
    "float64_missing": "float64",
    "float32_missing": "float32",
    "int64_complete": "int64",
    "Int64_missing": "Float64",
    "Float64_missing": "Float64",
    "Int64_complete": "Int64",
}


@pytest.mark.parametrize(
    "imputer",
    [MeanImputer(), MICEImputer(random_state=0), RegressionImputer()],
    ids=lambda i: type(i).__name__,
)
def test_dtypes_follow_the_policy(imputer: object, mixed: pd.DataFrame) -> None:
    result = imputer.impute(mixed)  # type: ignore[attr-defined]

    assert {c: str(result[c].dtype) for c in result.columns} == EXPECTED
    assert result.notna().all().all()
    for column in ("int64_complete", "Int64_complete"):
        pd.testing.assert_series_equal(result[column], mixed[column])


def test_values_left_missing_stay_na_in_nullable_columns() -> None:
    df = pd.DataFrame({"a": pd.array([None, 1, None, 3], dtype="Int64")})

    result = LOCFImputer().impute(df)

    assert str(result["a"].dtype) == "Float64"
    assert result["a"].isna().tolist() == [True, False, False, False]
    assert result["a"].iloc[0] is pd.NA
    assert result["a"].iloc[1:].tolist() == [1.0, 1.0, 3.0]


def test_duplicate_index_labels_do_not_misalign_columns() -> None:
    df = pd.DataFrame(
        {"a": [1.0, np.nan, 3.0, 4.0], "b": pd.array([10, 20, 30, 40], dtype="Int64")},
        index=[0, 0, 1, 1],
    )

    result = mean_impute(df)

    pd.testing.assert_index_equal(result.index, df.index)
    pd.testing.assert_series_equal(result["b"], df["b"])
    assert result["a"].tolist() == [1.0, 8.0 / 3.0, 3.0, 4.0]


def test_input_with_nullable_columns_is_not_modified() -> None:
    df = pd.DataFrame(
        {"a": pd.array([1, None, 3], dtype="Int64"), "b": [1.0, np.nan, 3.0]}
    )
    snapshot = df.copy()

    MICEImputer(random_state=0).impute(df)

    pd.testing.assert_frame_equal(df, snapshot)
