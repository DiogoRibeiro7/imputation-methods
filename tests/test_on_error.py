"""on_error: failures raise, fall back explicitly, or fall back with a FutureWarning."""

from __future__ import annotations

import inspect
import logging
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

import imputation_methods as im
from imputation_methods import BaseImputer, ImputationError


def _raiser(error: Exception) -> Callable[..., Any]:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise error

    return fail


class _FailingEstimator:
    """Stands in for a scikit-learn estimator whose ``fit`` always fails."""

    error: Exception = ValueError("model could not be fitted")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def fit(self, *args: Any, **kwargs: Any) -> _FailingEstimator:
        raise self.error


def _patch_instance(attribute: str, error: Exception) -> Callable[..., None]:
    def apply(imputer: BaseImputer, monkeypatch: pytest.MonkeyPatch) -> None:
        owner_name, method = attribute.split(".")
        monkeypatch.setattr(getattr(imputer, owner_name), method, _raiser(error))

    return apply


def _patch_module(target: str, replacement: Any) -> Callable[..., None]:
    def apply(imputer: BaseImputer, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(target, replacement)

    return apply


# imputer name -> (constructor kwargs, how to make it fail, expected fallback)
CASES: dict[str, tuple[dict[str, Any], Callable[..., None], type[BaseImputer]]] = {
    "MICEImputer": (
        {"random_state": 0},
        _patch_instance("_imputer.fit_transform", ValueError("singular")),
        im.MeanImputer,
    ),
    "MissForestImputer": (
        {"random_state": 0},
        _patch_instance("_imputer.fit_transform", ValueError("singular")),
        im.MedianImputer,
    ),
    "KNNImputer": (
        {},
        _patch_instance("_imputer.fit_transform", ValueError("bad input")),
        im.MeanImputer,
    ),
    "AutoencoderImputer": (
        {"max_iter": 5},
        _patch_instance("_model.fit", ValueError("bad input")),
        im.MeanImputer,
    ),
    "SoftImputeImputer": (
        {},
        _patch_module(
            "imputation_methods.matrix._soft_impute",
            _raiser(np.linalg.LinAlgError("SVD did not converge")),
        ),
        im.MeanImputer,
    ),
    "PPCAImputer": (
        {},
        _patch_module(
            "imputation_methods.matrix._ppca_impute",
            _raiser(np.linalg.LinAlgError("singular matrix")),
        ),
        im.MeanImputer,
    ),
    "RegressionImputer": (
        {},
        _patch_module(
            "imputation_methods.regression.LinearRegression", _FailingEstimator
        ),
        im.MeanImputer,
    ),
    "PMMImputer": (
        {"random_state": 0},
        _patch_module(
            "imputation_methods.regression.LinearRegression", _FailingEstimator
        ),
        im.MeanImputer,
    ),
    "GaussianProcessImputer": (
        {},
        _patch_module(
            "imputation_methods.regression.GaussianProcessRegressor", _FailingEstimator
        ),
        im.MeanImputer,
    ),
    "RANSACImputer": (
        {"random_state": 0},
        _patch_module(
            "imputation_methods.regression.RANSACRegressor", _FailingEstimator
        ),
        im.MedianImputer,
    ),
    "RadiusNeighborsImputer": (
        {},
        _patch_module(
            "imputation_methods.neighbors.RadiusNeighborsRegressor", _FailingEstimator
        ),
        im.MeanImputer,
    ),
}
NAMES = sorted(CASES)


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    frame.loc[[3, 10, 21], "a"] = np.nan
    frame.loc[[5, 17], "b"] = np.nan
    frame.loc[[8, 25], "c"] = np.nan
    return frame


def _failing_imputer(
    name: str, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> BaseImputer:
    base_kwargs, make_fail, _ = CASES[name]
    imputer = getattr(im, name)(**base_kwargs, **kwargs)
    make_fail(imputer, monkeypatch)
    return imputer


def test_every_imputer_with_on_error_is_covered() -> None:
    with_on_error = {
        name
        for name in im.__all__
        if inspect.isclass(getattr(im, name))
        and issubclass(getattr(im, name), BaseImputer)
        and "on_error" in inspect.signature(getattr(im, name)).parameters
    }
    assert with_on_error == set(CASES)


@pytest.mark.parametrize("name", NAMES)
def test_raise_reports_the_failure(
    name: str, df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    imputer = _failing_imputer(name, monkeypatch, on_error="raise")

    with pytest.raises(
        ImputationError, match=rf"{name} failed.*on_error='fallback'"
    ) as info:
        imputer.impute(df)

    assert isinstance(info.value, RuntimeError)
    assert info.value.__cause__ is not None


@pytest.mark.parametrize("name", NAMES)
def test_fallback_uses_the_simpler_method_quietly(
    name: str,
    df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    imputer = _failing_imputer(name, monkeypatch, on_error="fallback")
    expected = CASES[name][2]().impute(df)

    with (
        caplog.at_level(logging.WARNING, logger="imputation_methods"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error", FutureWarning)
        result = imputer.impute(df)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    assert any(f"{name} failed" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("name", NAMES)
def test_default_falls_back_with_a_future_warning(
    name: str, df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    imputer = _failing_imputer(name, monkeypatch)
    expected = CASES[name][2]().impute(df)

    with pytest.warns(
        FutureWarning, match=r"default will change to on_error='raise'"
    ) as record:
        result = imputer.impute(df)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    # The warning points at the caller's impute() call, not at library internals.
    assert all(w.filename == __file__ for w in record)


def test_unexpected_errors_fall_back_to_the_median(
    df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken(_FailingEstimator):
        error = RuntimeError("something unexpected")

    monkeypatch.setattr("imputation_methods.regression.LinearRegression", Broken)
    result = im.RegressionImputer(on_error="fallback").impute(df)
    pd.testing.assert_frame_equal(result, im.MedianImputer().impute(df))


@pytest.mark.parametrize("name", NAMES)
def test_invalid_on_error_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="on_error must be"):
        getattr(im, name)(**CASES[name][0], on_error="ignore")


def test_shortcuts_forward_on_error(
    df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "imputation_methods.regression.LinearRegression", _FailingEstimator
    )
    with pytest.raises(ImputationError):
        im.regression_impute(df, on_error="raise")
    result = im.regression_impute(df, on_error="fallback")
    pd.testing.assert_frame_equal(result, im.MeanImputer().impute(df))


def test_warning_through_a_shortcut_points_at_the_caller(
    df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "imputation_methods.regression.LinearRegression", _FailingEstimator
    )
    with pytest.warns(FutureWarning) as record:
        im.regression_impute(df)
    assert record
    assert all(w.filename == __file__ for w in record)


def test_hybrid_moves_on_when_a_member_raises(
    df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "imputation_methods.regression.LinearRegression", _FailingEstimator
    )
    hybrid = im.HybridImputer(
        methods=[im.RegressionImputer(on_error="raise"), im.MedianImputer()]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = hybrid.impute(df)
    pd.testing.assert_frame_equal(result, im.MedianImputer().impute(df))


def test_successful_imputations_do_not_warn(df: pd.DataFrame) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        im.RegressionImputer().impute(df)
        im.MICEImputer(random_state=0).impute(df)
