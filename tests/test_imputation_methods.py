import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
import pytest

from src.imputation_methods import (
    MeanImputer,
    MedianImputer,
    KNNImputerMethod,
    PMMImputer,
    MICEImputer,
    RegressionImputer,
    StochasticRegressionImputer,
    LOCFImputer,
    NOCBImputer,
)


def test_mean_impute_basic():
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, np.nan, 7, 8]})
    imputed = MeanImputer().impute(df)
    assert not imputed.isna().any().any()
    assert np.isclose(imputed.loc[2, "a"], (1 + 2 + 4) / 3)


def test_median_impute_basic():
    df = pd.DataFrame({"a": [1, 2, np.nan, 10]})
    imputed = MedianImputer().impute(df)
    assert not imputed.isna().any().any()
    assert imputed.loc[2, "a"] == 2


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


def test_mice_impute_basic():
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, np.nan, 7, 8]})
    imputed = MICEImputer(random_state=0).impute(df)
    assert not imputed.isna().any().any()


def test_non_numeric_raises():
    df = pd.DataFrame({"a": [1, 2, np.nan], "b": ["x", "y", "z"]})
    with pytest.raises(TypeError):
        MeanImputer().impute(df)


def test_no_missing():
    df = pd.DataFrame({"a": [1, 2, 3]})
    imputed = MeanImputer().impute(df)
    pd.testing.assert_frame_equal(imputed, df)


def test_all_missing():
    df = pd.DataFrame({"a": [np.nan, np.nan]})
    imputed = MeanImputer().impute(df)
    assert imputed.isna().all().all()


def test_single_column():
    df = pd.DataFrame({"a": [1, np.nan, 3]})
    imputed = MedianImputer().impute(df)
    assert not imputed.isna().any().any()


def test_regression_imputer():
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, 6, 7, 8]})
    imputed = RegressionImputer().impute(df)
    assert not imputed.isna().any().any()


def test_stochastic_regression_imputer():
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, 6, 7, 8]})
    imputed = StochasticRegressionImputer(random_state=0).impute(df)
    assert not imputed.isna().any().any()


def test_locf_nocb():
    df = pd.DataFrame({"a": [np.nan, 1, np.nan, 3, np.nan]})
    locf = LOCFImputer().impute(df)
    assert pd.isna(locf.loc[0, "a"])
    assert locf.loc[2, "a"] == 1
    nocb = NOCBImputer().impute(df)
    assert nocb.loc[0, "a"] == 1
    assert nocb.loc[2, "a"] == 3
