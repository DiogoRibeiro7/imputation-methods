"""Tests for the NumPy SoftImpute and probabilistic PCA implementations."""

import numpy as np
import pandas as pd
import pytest

from imputation_methods import (
    BayesianPCAImputer,
    MeanImputer,
    SoftImputeImputer,
    bayesian_pca_impute,
    soft_impute,
)


def _low_rank_frame(
    n_rows: int = 200, n_cols: int = 8, rank: int = 2, missing_rate: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(n_rows, rank)) @ rng.normal(size=(rank, n_cols))
    values += 0.05 * rng.normal(size=values.shape) + rng.normal(size=n_cols) * 3
    complete = pd.DataFrame(values, columns=[f"x{i}" for i in range(n_cols)])
    mask = rng.random(complete.shape) < missing_rate
    return complete, complete.mask(mask), mask


def _rmse(complete: pd.DataFrame, imputed: pd.DataFrame, mask: np.ndarray) -> float:
    diff = imputed.to_numpy()[mask] - complete.to_numpy()[mask]
    return float(np.sqrt(np.mean(diff**2)))


@pytest.mark.parametrize(
    ("imputer", "max_ratio"),
    [
        (SoftImputeImputer(), 0.7),
        (SoftImputeImputer(init_fill_method="mean"), 0.7),
        (BayesianPCAImputer(n_components=2), 0.25),
    ],
    ids=["softimpute-zero", "softimpute-mean", "ppca"],
)
def test_beats_mean_imputation_on_low_rank_data(
    imputer: SoftImputeImputer | BayesianPCAImputer, max_ratio: float
) -> None:
    complete, missing, mask = _low_rank_frame()
    baseline = _rmse(complete, MeanImputer().impute(missing), mask)
    result = imputer.impute(missing)

    assert result.notna().all().all()
    assert _rmse(complete, result, mask) < max_ratio * baseline


@pytest.mark.parametrize("imputer_cls", [SoftImputeImputer, BayesianPCAImputer])
def test_observed_values_and_input_are_unchanged(imputer_cls: type) -> None:
    _, missing, mask = _low_rank_frame(n_rows=40, n_cols=4)
    snapshot = missing.copy()

    result = imputer_cls().impute(missing)

    pd.testing.assert_frame_equal(missing, snapshot)
    np.testing.assert_allclose(result.to_numpy()[~mask], missing.to_numpy()[~mask])
    pd.testing.assert_index_equal(result.index, missing.index)
    pd.testing.assert_index_equal(result.columns, missing.columns)


@pytest.mark.parametrize("imputer_cls", [SoftImputeImputer, BayesianPCAImputer])
def test_is_deterministic(imputer_cls: type) -> None:
    _, missing, _ = _low_rank_frame(n_rows=40, n_cols=4)
    pd.testing.assert_frame_equal(
        imputer_cls().impute(missing), imputer_cls().impute(missing)
    )


@pytest.mark.parametrize("imputer_cls", [SoftImputeImputer, BayesianPCAImputer])
def test_fully_missing_column_stays_missing(imputer_cls: type) -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "b": [np.nan] * 5,
            "c": [2.0, np.nan, 6.0, 8.0, 10.0],
        }
    )
    result = imputer_cls().impute(df)
    assert result["b"].isna().all()
    assert result[["a", "c"]].notna().all().all()


def test_ppca_instance_can_be_reused_on_different_shapes() -> None:
    _, missing, _ = _low_rank_frame(n_rows=40, n_cols=5)
    imputer = BayesianPCAImputer(n_components=2)
    imputer.impute(missing)
    result = imputer.impute(missing.iloc[:, :3])
    assert result.shape == (40, 3)
    assert result.notna().all().all()


def test_ppca_columns_below_min_obs_are_mean_imputed() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0],
            "b": [2.0, 4.0, np.nan, 8.0, 10.0, 12.0],
            "sparse": [np.nan, np.nan, 7.0, np.nan, np.nan, np.nan],
        }
    )
    result = BayesianPCAImputer(min_obs=2).impute(df)
    assert (result["sparse"] == 7.0).all()
    assert result.notna().all().all()


def test_accepts_nullable_integer_columns() -> None:
    df = pd.DataFrame(
        {
            "a": pd.array([1, None, 3, 4, 5], dtype="Int64"),
            "b": pd.array([2, 4, 6, None, 10], dtype="Int64"),
        }
    )
    assert SoftImputeImputer().impute(df).notna().all().all()
    assert BayesianPCAImputer().impute(df).notna().all().all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"init_fill_method": "random"}, "init_fill_method"),
        ({"max_iters": 0}, "max_iters"),
        ({"shrinkage_value": -1.0}, "shrinkage_value"),
    ],
)
def test_softimpute_validates_arguments(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SoftImputeImputer(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_components": 0}, "n_components"),
        ({"min_obs": 0}, "min_obs"),
        ({"max_iter": 0}, "max_iter"),
    ],
)
def test_ppca_validates_arguments(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BayesianPCAImputer(**kwargs)


def test_functional_wrappers() -> None:
    _, missing, _ = _low_rank_frame(n_rows=30, n_cols=4)
    assert soft_impute(missing, max_iters=10).notna().all().all()
    assert bayesian_pca_impute(missing, n_components=2).notna().all().all()
