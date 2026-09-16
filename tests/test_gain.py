"""Tests for the NumPy GAIN implementation."""

import numpy as np
import pandas as pd
import pytest

from imputation_methods import GAINImputer, MeanImputer, gain_impute
from imputation_methods.neural import _MLP


def _low_rank_frame(
    n_rows: int = 400, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_rows, 2)) @ rng.normal(size=(2, 6))
    values += 0.1 * rng.normal(size=values.shape)
    complete = pd.DataFrame(values, columns=[f"x{i}" for i in range(6)])
    mask = rng.random(complete.shape) < 0.2
    return complete, complete.mask(mask), mask


def _rmse(complete: pd.DataFrame, imputed: pd.DataFrame, mask: np.ndarray) -> float:
    diff = imputed.to_numpy()[mask] - complete.to_numpy()[mask]
    return float(np.sqrt(np.mean(diff**2)))


def test_mlp_gradients_match_finite_differences() -> None:
    rng = np.random.default_rng(0)
    net = _MLP([6, 4, 4, 3], rng)
    # Non-zero biases keep pre-activations away from the ReLU kink at 0.
    for k in range(1, len(net.params), 2):
        net.params[k] = rng.normal(scale=0.1, size=net.params[k].shape)
    x = rng.normal(size=(5, 6))
    target = (rng.random((5, 3)) < 0.5).astype(float)

    def loss(inputs: np.ndarray) -> float:
        prob, _ = net.forward(inputs)
        return float(-np.mean(target * np.log(prob) + (1 - target) * np.log(1 - prob)))

    prob, activations = net.forward(x)
    grads, grad_input = net.backward(activations, (prob - target) / prob.size)

    eps = 1e-6
    for param, grad in zip(net.params, grads, strict=True):
        numeric = np.zeros_like(param)
        for i in np.ndindex(param.shape):
            original = param[i]
            param[i] = original + eps
            upper = loss(x)
            param[i] = original - eps
            lower = loss(x)
            param[i] = original
            numeric[i] = (upper - lower) / (2 * eps)
        np.testing.assert_allclose(grad, numeric, atol=1e-7)

    numeric_input = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        shifted = x.copy()
        shifted[i] += eps
        upper = loss(shifted)
        shifted[i] -= 2 * eps
        numeric_input[i] = (upper - loss(shifted)) / (2 * eps)
    np.testing.assert_allclose(grad_input, numeric_input, atol=1e-7)


def test_beats_mean_imputation_on_correlated_data() -> None:
    complete, missing, mask = _low_rank_frame()
    baseline = _rmse(complete, MeanImputer().impute(missing), mask)

    result = GAINImputer(iterations=1500, random_state=0).impute(missing)

    assert result.notna().all().all()
    assert _rmse(complete, result, mask) < 0.75 * baseline


def test_is_reproducible_with_random_state() -> None:
    _, missing, _ = _low_rank_frame(n_rows=60)
    first = GAINImputer(iterations=50, random_state=3).impute(missing)
    second = gain_impute(missing, iterations=50, random_state=3)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_give_different_imputations() -> None:
    _, missing, mask = _low_rank_frame(n_rows=60)
    first = GAINImputer(iterations=50, random_state=1).impute(missing)
    second = GAINImputer(iterations=50, random_state=2).impute(missing)
    assert not np.allclose(first.to_numpy()[mask], second.to_numpy()[mask])


def test_imputations_stay_within_observed_range() -> None:
    _, missing, _ = _low_rank_frame(n_rows=100)
    result = GAINImputer(iterations=200, random_state=0).impute(missing)
    low, high = missing.min() - 1e-5, missing.max() + 1e-5
    assert ((result >= low) & (result <= high)).all().all()


def test_constant_column_is_supported() -> None:
    df = pd.DataFrame({"a": [5.0, 5.0, np.nan, 5.0], "b": [1.0, np.nan, 3.0, 4.0]})
    result = GAINImputer(iterations=50, random_state=0).impute(df)
    np.testing.assert_allclose(result["a"], 5.0, atol=1e-5)
    assert result["b"].notna().all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"hint_rate": 1.5}, "hint_rate"),
        ({"alpha": -1.0}, "alpha"),
        ({"iterations": 0}, "iterations"),
        ({"learning_rate": 0.0}, "learning_rate"),
    ],
)
def test_validates_arguments(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GAINImputer(**kwargs)
