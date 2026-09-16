"""The output dtype policy shared by every imputer.

* A column without missing values is returned unchanged: same values, same dtype.
* A column that had missing values is returned as floating point, because imputed
  values are generally not integers. NumPy float columns keep their precision
  (``float32`` stays ``float32``); pandas nullable columns (``Int64``, ``Float64``,
  ...) become ``Float64``, so any values left missing stay ``<NA>``.
* Columns an imputer adds (e.g. ``IndicatorImputer``'s indicators) and non-numeric
  columns (e.g. ``GroupMeanImputer``'s grouping column) are returned as produced.

Imputers work on a copy in which nullable numeric columns are converted to
``float64`` with ``NaN``, so the algorithms only ever see NumPy dtypes.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import pandas as pd

F = TypeVar("F", bound=Callable[..., pd.DataFrame])


def _is_nullable_numeric(series: pd.Series) -> bool:
    return isinstance(
        series.dtype, pd.api.extensions.ExtensionDtype
    ) and pd.api.types.is_numeric_dtype(series.dtype)


def _needs_float64(series: pd.Series) -> bool:
    if _is_nullable_numeric(series):
        return True
    return (
        pd.api.types.is_numeric_dtype(series.dtype)
        and series.dtype != np.float64
        and bool(series.isna().any())
    )


def _working_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with nullable and incomplete numeric columns as ``float64``.

    Imputed values are written into incomplete columns, so those must hold full
    ``float64`` values while the algorithm runs (pandas refuses lossy writes into,
    e.g., ``float32``). The dtype is restored afterwards by ``_restore_dtypes``.
    """
    convert = [column for column in df.columns if _needs_float64(df[column])]
    if not convert:
        return df
    working = df.copy()
    for column in convert:
        working[column] = df[column].to_numpy(dtype="float64", na_value=np.nan)
    return working


def _restore_dtypes(result: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """Apply the output dtype policy to ``result``, given the caller's input."""
    for column in original.columns:
        if column not in result.columns:
            continue
        source = original[column]
        if not pd.api.types.is_numeric_dtype(source.dtype):
            continue
        if not source.isna().any():
            result[column] = source.array
        elif isinstance(source.dtype, pd.api.extensions.ExtensionDtype):
            result[column] = result[column].astype("Float64")
        elif result[column].dtype != source.dtype:
            result[column] = result[column].astype(source.dtype)
    return result


def preserve_dtypes(impute: F) -> F:
    """Apply the output dtype policy to an imputer's ``impute`` method."""

    @functools.wraps(impute)
    def wrapper(self: Any, df: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            # Let the imputer's own validation report the problem.
            return impute(self, df, *args, **kwargs)
        result = impute(self, _working_copy(df), *args, **kwargs)
        return _restore_dtypes(result, df)

    return wrapper  # type: ignore[return-value]
