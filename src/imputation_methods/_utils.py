"""Private helpers shared by several imputer modules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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
