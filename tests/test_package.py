"""Tests for the package surface: exports, version and shared input validation."""

import inspect
import re

import numpy as np
import pandas as pd
import pytest

import imputation_methods
from imputation_methods import BaseImputer, MeanImputer


def test_version_is_pep440() -> None:
    assert re.match(r"^\d+\.\d+\.\d+", imputation_methods.__version__)


def test_all_exports_resolve() -> None:
    for name in imputation_methods.__all__:
        assert hasattr(imputation_methods, name), name


def test_all_has_no_duplicates() -> None:
    names = imputation_methods.__all__
    assert len(names) == len(set(names))


def _imputer_classes() -> list[type[BaseImputer]]:
    return [
        obj
        for obj in vars(imputation_methods).values()
        if inspect.isclass(obj)
        and issubclass(obj, BaseImputer)
        and obj is not BaseImputer
    ]


def test_every_imputer_is_exported() -> None:
    classes = _imputer_classes()
    assert len(classes) >= 40
    for cls in classes:
        assert cls.__name__ in imputation_methods.__all__


@pytest.mark.parametrize("cls", _imputer_classes(), ids=lambda c: c.__name__)
def test_every_imputer_has_docstring(cls: type[BaseImputer]) -> None:
    assert cls.__doc__
    assert cls.impute.__doc__


def test_rejects_non_dataframe_input() -> None:
    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        MeanImputer().impute(np.array([[1.0, np.nan]]))  # type: ignore[arg-type]


def test_non_numeric_error_names_columns() -> None:
    df = pd.DataFrame({"x": [1.0, np.nan], "label": ["a", "b"]})
    with pytest.raises(TypeError, match="label"):
        MeanImputer().impute(df)
