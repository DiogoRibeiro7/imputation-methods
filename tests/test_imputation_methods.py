import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
import pytest

from src.imputation_methods import (
    MeanImputer,
    KNNImputerMethod,
    PMMImputer,
)


def test_mean_impute_basic():
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, np.nan, 7, 8]})
    imputed = MeanImputer().impute(df)
    assert not imputed.isna().any().any()
    assert np.isclose(imputed.loc[2, "a"], (1 + 2 + 4) / 3)


def test_knn_impute_basic():
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, np.nan, 7, 8]})
    imputed = KNNImputerMethod(k=2).impute(df)
    assert not imputed.isna().any().any()


def test_pmm_impute_basic():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, np.nan],
        "y": [5, 6, 7, 8, 9],
    })
    imputed = PMMImputer(k=2, random_state=0).impute(df)
    assert not imputed.isna().any().any()


def test_non_numeric_raises():
    df = pd.DataFrame({"a": [1, 2, np.nan], "b": ["x", "y", "z"]})
    with pytest.raises(TypeError):
        MeanImputer().impute(df)
