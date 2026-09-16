"""Tests for the functional ``*_impute`` shortcuts."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

import imputation_methods
from imputation_methods import BaseImputer, functional


def _wrappers() -> list[tuple[Callable[..., pd.DataFrame], type[BaseImputer]]]:
    pairs = []
    for _, function in inspect.getmembers(functional, inspect.isfunction):
        if function.__module__ != functional.__name__:
            continue
        match = re.search(r":class:`(\w+)`", function.__doc__ or "")
        assert match, f"{function.__name__} doesn't name the class it wraps"
        pairs.append((function, getattr(imputation_methods, match.group(1))))
    return pairs


WRAPPERS = _wrappers()
IDS = [function.__name__ for function, _ in WRAPPERS]

# Arguments needed to build some imputers, or to keep the test fast.
KWARGS: dict[str, dict[str, object]] = {
    "group_mean_impute": {"group_col": "g"},
    "gain_impute": {"max_iter": 50},
    "autoencoder_impute": {"max_iter": 50},
}


def test_every_imputer_has_a_shortcut() -> None:
    wrapped = {cls for _, cls in WRAPPERS}
    imputers = {
        obj
        for obj in vars(imputation_methods).values()
        if inspect.isclass(obj)
        and issubclass(obj, BaseImputer)
        and obj is not BaseImputer
    }
    assert imputers == wrapped


@pytest.mark.parametrize(("function", "cls"), WRAPPERS, ids=IDS)
def test_shortcut_accepts_the_same_parameters_as_its_class(
    function: Callable[..., pd.DataFrame], cls: type[BaseImputer]
) -> None:
    function_params = dict(inspect.signature(function).parameters)
    assert next(iter(function_params)) == "df"
    del function_params["df"]
    class_params = inspect.signature(cls).parameters

    assert set(function_params) == set(class_params)
    for name, parameter in function_params.items():
        assert parameter.default == class_params[name].default, name


@pytest.mark.parametrize(("function", "cls"), WRAPPERS, ids=IDS)
def test_shortcut_matches_its_class(
    function: Callable[..., pd.DataFrame], cls: type[BaseImputer]
) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "g": np.repeat([0, 1], 15),
            "a": rng.normal(size=30),
            "b": rng.normal(size=30),
        }
    )
    df.loc[[3, 11, 20], "a"] = np.nan
    df.loc[[5, 17, 25], "b"] = np.nan

    kwargs = dict(KWARGS.get(function.__name__, {}))
    if "random_state" in inspect.signature(cls).parameters:
        kwargs["random_state"] = 0

    pd.testing.assert_frame_equal(function(df, **kwargs), cls(**kwargs).impute(df))
