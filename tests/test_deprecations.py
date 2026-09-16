"""Renamed API elements keep working, with a FutureWarning, until 1.0.0."""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import numpy as np
import pandas as pd
import pytest

import imputation_methods as im


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {"g": np.repeat([0, 1], 10), "a": rng.normal(size=20), "b": rng.normal(size=20)}
    )
    frame.loc[[2, 9, 15], "a"] = np.nan
    frame.loc[[4, 12], "b"] = np.nan
    return frame


RENAMED_ATTRIBUTES = [
    ("imputation_methods", "KNNImputerMethod", "KNNImputer"),
    ("imputation_methods", "BayesianPCAImputer", "PPCAImputer"),
    ("imputation_methods", "predictive_mean_matching", "pmm_impute"),
    ("imputation_methods", "bayesian_pca_impute", "ppca_impute"),
    ("imputation_methods.neighbors", "KNNImputerMethod", "KNNImputer"),
    ("imputation_methods.matrix", "BayesianPCAImputer", "PPCAImputer"),
    ("imputation_methods.functional", "predictive_mean_matching", "pmm_impute"),
    ("imputation_methods.functional", "bayesian_pca_impute", "ppca_impute"),
]


@pytest.mark.parametrize(("module_name", "old", "new"), RENAMED_ATTRIBUTES)
def test_old_names_resolve_to_new_objects_with_warning(
    module_name: str, old: str, new: str
) -> None:
    module = importlib.import_module(module_name)
    with pytest.warns(
        FutureWarning, match=rf"imputation-methods: .*\b{old} is deprecated"
    ):
        resolved = getattr(module, old)
    assert resolved is getattr(module, new)


def test_from_import_of_old_name_works() -> None:
    with pytest.warns(FutureWarning, match="KNNImputerMethod is deprecated"):
        from imputation_methods import KNNImputerMethod
    assert KNNImputerMethod is im.KNNImputer


@pytest.mark.parametrize("old", sorted({old for _, old, _ in RENAMED_ATTRIBUTES}))
def test_old_names_are_hidden_from_public_listings(old: str) -> None:
    assert old not in im.__all__
    assert old not in vars(im)


def test_unknown_attribute_still_raises() -> None:
    with pytest.raises(AttributeError, match="no attribute 'NotAnImputer'"):
        _ = im.NotAnImputer  # type: ignore[attr-defined]


RENAMED_PARAMETERS: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    ("KNNImputer", {"k": 2}, {"n_neighbors": 2}),
    ("PMMImputer", {"k": 2, "random_state": 0}, {"n_neighbors": 2, "random_state": 0}),
    ("EndOfDistributionImputer", {"k": 2.0}, {"n_std": 2.0}),
    (
        "GroupMeanImputer",
        {"group_col": "g", "method": "median"},
        {"group_col": "g", "strategy": "median"},
    ),
    ("MovingAverageImputer", {"method": "median"}, {"strategy": "median"}),
    ("SeasonalImputer", {"method": "mean"}, {"strategy": "mean"}),
    ("SoftImputeImputer", {"max_iters": 5}, {"max_iter": 5}),
    (
        "GAINImputer",
        {"iterations": 20, "random_state": 0},
        {"max_iter": 20, "random_state": 0},
    ),
    ("knn_impute", {"k": 2}, {"n_neighbors": 2}),
    ("pmm_impute", {"k": 2, "random_state": 0}, {"n_neighbors": 2, "random_state": 0}),
    ("end_of_distribution_impute", {"k": 2.0}, {"n_std": 2.0}),
    (
        "group_mean_impute",
        {"group_col": "g", "method": "median"},
        {"group_col": "g", "strategy": "median"},
    ),
    ("moving_average_impute", {"method": "median"}, {"strategy": "median"}),
    ("seasonal_impute", {"method": "mean"}, {"strategy": "mean"}),
    ("soft_impute", {"max_iters": 5}, {"max_iter": 5}),
    (
        "gain_impute",
        {"iterations": 20, "random_state": 0},
        {"max_iter": 20, "random_state": 0},
    ),
]


def _run(name: str, df: pd.DataFrame, kwargs: dict[str, Any]) -> pd.DataFrame:
    target = getattr(im, name)
    if inspect.isclass(target):
        return target(**kwargs).impute(df)
    return target(df, **kwargs)


@pytest.mark.parametrize(
    ("name", "old_kwargs", "new_kwargs"),
    RENAMED_PARAMETERS,
    ids=[
        f"{name}-{next(iter(set(old) - set(new)))}"
        for name, old, new in RENAMED_PARAMETERS
    ],
)
def test_old_parameter_names_warn_and_give_the_same_result(
    name: str, old_kwargs: dict[str, Any], new_kwargs: dict[str, Any], df: pd.DataFrame
) -> None:
    (old,) = set(old_kwargs) - set(new_kwargs)
    (new,) = set(new_kwargs) - set(old_kwargs)

    with pytest.warns(
        FutureWarning, match=rf"\({old}=\.\.\.\) is deprecated .* use {new}="
    ):
        result = _run(name, df, old_kwargs)

    pd.testing.assert_frame_equal(result, _run(name, df, new_kwargs))


@pytest.mark.parametrize(
    ("name", "old_kwargs", "new_kwargs"),
    RENAMED_PARAMETERS,
    ids=[name for name, _, _ in RENAMED_PARAMETERS],
)
def test_passing_old_and_new_parameter_names_is_an_error(
    name: str, old_kwargs: dict[str, Any], new_kwargs: dict[str, Any], df: pd.DataFrame
) -> None:
    with pytest.raises(TypeError, match="deprecated"):
        _run(name, df, {**old_kwargs, **new_kwargs})


@pytest.mark.parametrize(
    "name", sorted({name for name, _, _ in RENAMED_PARAMETERS}), ids=str
)
def test_signatures_only_show_new_parameter_names(name: str) -> None:
    parameters = inspect.signature(getattr(im, name)).parameters
    assert not {"k", "method", "max_iters", "iterations"} & set(parameters)


@pytest.mark.parametrize(
    ("cls_name", "old", "new", "value"),
    [
        ("KNNImputer", "k", "n_neighbors", 3),
        ("PMMImputer", "k", "n_neighbors", 3),
        ("EndOfDistributionImputer", "k", "n_std", 2.5),
        ("MovingAverageImputer", "method", "strategy", "median"),
        ("SeasonalImputer", "method", "strategy", "mean"),
        ("SoftImputeImputer", "max_iters", "max_iter", 7),
        ("GAINImputer", "iterations", "max_iter", 7),
    ],
)
def test_values_passed_under_old_names_are_stored_under_new_names(
    cls_name: str, old: str, new: str, value: object
) -> None:
    with pytest.warns(FutureWarning):
        imputer = getattr(im, cls_name)(**{old: value})
    assert getattr(imputer, new) == value
    assert not hasattr(imputer, old)
