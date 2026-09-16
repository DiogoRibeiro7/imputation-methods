"""Metrics for scoring imputations against known ground truth.

Score only the entries that were imputed. Comparing whole dataframes also counts
the observed cells, which match exactly and make every method look better than it
is. The usual pattern is to hide values in complete data with a boolean mask, then
compare the masked positions:

Examples:
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imputation_methods import MeanImputer, mae, rmse
    >>> complete = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    >>> mask = np.array([[False, False], [True, False], [False, True], [False, False]])
    >>> imputed = MeanImputer().impute(complete.mask(mask))
    >>> true = pd.Series(complete.to_numpy()[mask])
    >>> pred = pd.Series(imputed.to_numpy()[mask])
    >>> round(rmse(true, pred), 4), round(mae(true, pred), 4)
    (1.0541, 1.0)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(true: pd.Series, pred: pd.Series) -> float:
    """Calculate the root mean squared error between true and imputed values.

    Args:
        true: Ground-truth values for the imputed entries.
        pred: Imputed values, aligned with ``true``.

    Returns:
        The RMSE. ``NaN`` pairs are ignored, so check that ``pred`` has no
        missing values first.
    """
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(true: pd.Series, pred: pd.Series) -> float:
    """Calculate the mean absolute error between true and imputed values.

    Args:
        true: Ground-truth values for the imputed entries.
        pred: Imputed values, aligned with ``true``.

    Returns:
        The MAE. ``NaN`` pairs are ignored, so check that ``pred`` has no
        missing values first.
    """
    return float(np.mean(np.abs(true - pred)))
