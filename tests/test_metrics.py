"""Tests for the evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from imputation_methods import (
    mae,
    rmse,
)


def test_rmse_calculation() -> None:
    """Test RMSE evaluation metric."""
    true = pd.Series([1, 2, 3, 4, 5])
    pred = pd.Series([1.1, 2.1, 2.9, 4.2, 4.8])
    error = rmse(true, pred)
    expected = np.sqrt(np.mean([0.01, 0.01, 0.01, 0.04, 0.04]))
    assert np.isclose(error, expected)


def test_mae_calculation() -> None:
    """Test MAE evaluation metric."""
    true = pd.Series([1, 2, 3, 4, 5])
    pred = pd.Series([1.1, 2.1, 2.9, 4.2, 4.8])
    error = mae(true, pred)
    expected = np.mean([0.1, 0.1, 0.1, 0.2, 0.2])
    assert np.isclose(error, expected)
