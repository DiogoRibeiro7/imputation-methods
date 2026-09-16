"""Private helpers shared by several imputer modules."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Hashable
from typing import Any, Literal

import numpy as np
import pandas as pd

from .base import ImputationError

logger = logging.getLogger("imputation_methods")

OnError = Literal["raise", "fallback"] | None
"""How an imputer reacts when its model cannot be fitted; see ``check_on_error``."""

_ON_ERROR_VALUES = ("raise", "fallback", None)


def fit_transform_non_empty(imputer: Any, df: pd.DataFrame) -> pd.DataFrame:
    """Apply a scikit-learn imputer to the columns that have observed values.

    scikit-learn imputers silently drop columns with no observed values, which
    would change the output shape. Those columns are returned unchanged (all
    ``NaN``) instead, and the rest are imputed as usual.

    Args:
        imputer: Unfitted scikit-learn transformer with ``fit_transform``.
        df: Numeric dataframe with missing values.

    Returns:
        Float dataframe with the same index and columns as ``df``.
    """
    result = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    non_empty = df.columns[df.notna().any()]
    if len(non_empty) > 0:
        result[non_empty] = imputer.fit_transform(df[non_empty])
    return result


def observed_median(series: pd.Series) -> float:
    """Median of the observed values, or NaN if there are none.

    Unlike ``Series.median``, this never emits NumPy's "Mean of empty slice"
    warning for a column with no observed values.
    """
    return float(series.median()) if series.notna().any() else float("nan")


def check_on_error(on_error: OnError) -> OnError:
    """Validate an ``on_error`` argument.

    ``"raise"`` raises :class:`~imputation_methods.ImputationError` when the model
    fails; ``"fallback"`` uses a simpler method and logs a warning; ``None`` (the
    default until 1.0.0) falls back like ``"fallback"`` but also emits a
    ``FutureWarning``, because the default will become ``"raise"``.

    Raises:
        ValueError: If ``on_error`` is not one of the accepted values.
    """
    if on_error not in _ON_ERROR_VALUES:
        raise ValueError(
            f"on_error must be 'raise', 'fallback' or None, got {on_error!r}"
        )
    return on_error


def raise_or_fall_back(
    on_error: OnError,
    *,
    imputer: str,
    error: Exception,
    fallback: str,
    column: Hashable | None = None,
    stacklevel: int = 3,
) -> None:
    """Handle a model failure according to ``on_error``.

    Returns normally when the caller should apply its fallback.

    Args:
        on_error: The imputer's ``on_error`` setting.
        imputer: Name of the failing imputer, for messages.
        error: The exception raised by the model.
        fallback: Human-readable name of the fallback, e.g. ``"mean imputation"``.
        column: Column being imputed, if the failure is column-specific.
        stacklevel: Passed to :func:`warnings.warn`, so the warning points at the
            user's ``impute`` call.

    Raises:
        ImputationError: If ``on_error`` is ``"raise"``.
    """
    what = f"{imputer} failed" + (
        f" for column {column!r}" if column is not None else ""
    )
    if on_error == "raise":
        raise ImputationError(
            f"{what}: {error}. Pass on_error='fallback' to use {fallback} instead."
        ) from error
    logger.warning("%s (%s); using %s", what, error, fallback)
    if on_error is None:
        warnings.warn(
            f"imputation-methods: {what} ({error}), so {fallback} was used instead. "
            "Falling back silently is deprecated and the default will change to "
            "on_error='raise' in 1.0.0. Pass on_error='fallback' to keep this "
            "behavior, or on_error='raise' to get an error.",
            FutureWarning,
            stacklevel=stacklevel,
        )
