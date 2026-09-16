"""Functional shortcuts: one ``*_impute(df, ...)`` function per imputer.

Each function builds the corresponding imputer with the given parameters
and immediately applies it to ``df``.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.gaussian_process.kernels import RBF

from .base import BaseImputer
from .ensemble import BaggingImputer, HybridImputer, StackingImputer
from .iterative import EMImputer, MICEImputer, MissForestImputer
from .matrix import BayesianPCAImputer, SoftImputeImputer
from .neighbors import KNNImputerMethod, LocalMeanImputer, RadiusNeighborsImputer
from .neural import AutoencoderImputer, GAINImputer
from .regression import (
    BayesianRidgeImputer,
    GaussianProcessImputer,
    HuberImputer,
    PMMImputer,
    RANSACImputer,
    RegressionImputer,
    StochasticRegressionImputer,
)
from .sampling import (
    ColdDeckImputer,
    HotDeckImputer,
    RandomSamplingImputer,
    ReferenceValues,
)
from .statistical import (
    ConstantImputer,
    EndOfDistributionImputer,
    GroupMeanImputer,
    IndicatorImputer,
    MeanImputer,
    MedianImputer,
    ModeImputer,
    QuantileImputer,
    TrimmedMeanImputer,
)
from .time_series import (
    ForwardFillFallbackImputer,
    InterpolationImputer,
    KalmanFilterImputer,
    LinearTrendImputer,
    LOCFImputer,
    MovingAverageImputer,
    NOCBImputer,
    PolynomialTrendImputer,
    SeasonalImputer,
    WeightedMovingAverageImputer,
)


def mean_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`MeanImputer`."""
    return MeanImputer().impute(df)


def median_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`MedianImputer`."""
    return MedianImputer().impute(df)


def knn_impute(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`KNNImputerMethod`."""
    return KNNImputerMethod(k=k).impute(df)


def predictive_mean_matching(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`PMMImputer`."""
    return PMMImputer(k=k).impute(df)


def mice_impute(
    df: pd.DataFrame,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`MICEImputer`."""
    return MICEImputer(random_state=random_state).impute(df)


def regression_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`RegressionImputer`."""
    return RegressionImputer().impute(df)


def stochastic_regression_impute(
    df: pd.DataFrame, random_state: int | None = None
) -> pd.DataFrame:
    """Wrapper for :class:`StochasticRegressionImputer`."""
    return StochasticRegressionImputer(random_state=random_state).impute(df)


def locf_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`LOCFImputer`."""
    return LOCFImputer().impute(df)


def nocb_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`NOCBImputer`."""
    return NOCBImputer().impute(df)


def hot_deck_impute(
    df: pd.DataFrame,
    stratify_cols: list[str] | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Wrapper for :class:`HotDeckImputer`."""
    return HotDeckImputer(
        stratify_cols=stratify_cols,
        random_state=random_state,
    ).impute(df)


def miss_forest_impute(
    df: pd.DataFrame, random_state: int | None = None
) -> pd.DataFrame:
    """Wrapper for :class:`MissForestImputer`."""
    return MissForestImputer(random_state=random_state).impute(df)


def bayesian_pca_impute(
    df: pd.DataFrame,
    n_components: int | None = None,
    min_obs: int = 1,
) -> pd.DataFrame:
    """Wrapper for :class:`BayesianPCAImputer`."""
    return BayesianPCAImputer(
        n_components=n_components,
        min_obs=min_obs,
    ).impute(df)


def soft_impute(
    df: pd.DataFrame, max_iters: int = 100, init_fill_method: str = "zero"
) -> pd.DataFrame:
    """Wrapper for :class:`SoftImputeImputer`."""
    return SoftImputeImputer(
        max_iters=max_iters,
        init_fill_method=init_fill_method,
    ).impute(df)


def autoencoder_impute(
    df: pd.DataFrame,
    hidden_layer_sizes: tuple[int, ...] = (10,),
    max_iter: int = 200,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Wrapper for :class:`AutoencoderImputer`."""
    return AutoencoderImputer(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
    ).impute(df)


def gain_impute(
    df: pd.DataFrame,
    batch_size: int = 128,
    hint_rate: float = 0.9,
    alpha: float = 100.0,
    iterations: int = 10000,
    learning_rate: float = 0.001,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Wrapper for :class:`GAINImputer`."""
    return GAINImputer(
        batch_size=batch_size,
        hint_rate=hint_rate,
        alpha=alpha,
        iterations=iterations,
        learning_rate=learning_rate,
        random_state=random_state,
    ).impute(df)


def gaussian_process_impute(
    df: pd.DataFrame,
    kernel: RBF | None = None,
    alpha: float = 1e-10,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Wrapper for :class:`GaussianProcessImputer`."""
    return GaussianProcessImputer(
        kernel=kernel,
        alpha=alpha,
        random_state=random_state,
    ).impute(df)


def interpolation_impute(
    df: pd.DataFrame,
    method: str = "linear",
    order: int = 2,
    limit: int | None = None,
    limit_direction: Literal["forward", "backward", "both"] = "both",
) -> pd.DataFrame:
    """Wrapper for :class:`InterpolationImputer`."""
    return InterpolationImputer(
        method=method, order=order, limit=limit, limit_direction=limit_direction
    ).impute(df)


def em_impute(
    df: pd.DataFrame,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Wrapper for :class:`EMImputer`."""
    return EMImputer(max_iter=max_iter, tol=tol, random_state=random_state).impute(df)


def moving_average_impute(
    df: pd.DataFrame,
    window: int = 3,
    method: str = "mean",
    min_periods: int = 1,
    center: bool = False,
) -> pd.DataFrame:
    """Wrapper for :class:`MovingAverageImputer`."""
    return MovingAverageImputer(
        window=window, method=method, min_periods=min_periods, center=center
    ).impute(df)


def random_sampling_impute(
    df: pd.DataFrame, random_state: int | None = None
) -> pd.DataFrame:
    """Wrapper for :class:`RandomSamplingImputer`."""
    return RandomSamplingImputer(random_state=random_state).impute(df)


def indicator_impute(
    df: pd.DataFrame, strategy: str = "mean", indicator_prefix: str = "missing_"
) -> pd.DataFrame:
    """Wrapper for :class:`IndicatorImputer`."""
    return IndicatorImputer(
        strategy=strategy, indicator_prefix=indicator_prefix
    ).impute(df)


def seasonal_impute(
    df: pd.DataFrame, period: int = 7, method: str = "median"
) -> pd.DataFrame:
    """Wrapper for :class:`SeasonalImputer`."""
    return SeasonalImputer(period=period, method=method).impute(df)


def quantile_impute(df: pd.DataFrame, quantile: float = 0.5) -> pd.DataFrame:
    """Wrapper for :class:`QuantileImputer`."""
    return QuantileImputer(quantile=quantile).impute(df)


def forward_fill_fallback_impute(
    df: pd.DataFrame, fallback: str = "mean"
) -> pd.DataFrame:
    """Wrapper for :class:`ForwardFillFallbackImputer`."""
    return ForwardFillFallbackImputer(fallback=fallback).impute(df)


def mode_impute(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Wrapper for :class:`ModeImputer`."""
    return ModeImputer(dropna=dropna).impute(df)


def constant_impute(
    df: pd.DataFrame, fill_value: float | dict[str, float] = 0
) -> pd.DataFrame:
    """Wrapper for :class:`ConstantImputer`."""
    return ConstantImputer(fill_value=fill_value).impute(df)


def end_of_distribution_impute(
    df: pd.DataFrame, position: str = "high", k: float = 3.0
) -> pd.DataFrame:
    """Wrapper for :class:`EndOfDistributionImputer`."""
    return EndOfDistributionImputer(position=position, k=k).impute(df)


def group_mean_impute(
    df: pd.DataFrame, group_col: str, method: str = "mean", global_fallback: bool = True
) -> pd.DataFrame:
    """Wrapper for :class:`GroupMeanImputer`."""
    return GroupMeanImputer(
        group_col=group_col, method=method, global_fallback=global_fallback
    ).impute(df)


def weighted_moving_average_impute(
    df: pd.DataFrame, alpha: float = 0.5, min_periods: int = 1
) -> pd.DataFrame:
    """Wrapper for :class:`WeightedMovingAverageImputer`."""
    return WeightedMovingAverageImputer(alpha=alpha, min_periods=min_periods).impute(df)


def linear_trend_impute(df: pd.DataFrame, use_index: bool = False) -> pd.DataFrame:
    """Wrapper for :class:`LinearTrendImputer`."""
    return LinearTrendImputer(use_index=use_index).impute(df)


def polynomial_trend_impute(
    df: pd.DataFrame, degree: int = 2, use_index: bool = False
) -> pd.DataFrame:
    """Wrapper for :class:`PolynomialTrendImputer`."""
    return PolynomialTrendImputer(degree=degree, use_index=use_index).impute(df)


def kalman_filter_impute(
    df: pd.DataFrame,
    process_variance: float = 1.0,
    measurement_variance: float = 1.0,
    initial_state: float | None = None,
    initial_covariance: float = 1.0,
) -> pd.DataFrame:
    """Wrapper for :class:`KalmanFilterImputer`."""
    return KalmanFilterImputer(
        process_variance=process_variance,
        measurement_variance=measurement_variance,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
    ).impute(df)


def cold_deck_impute(
    df: pd.DataFrame,
    reference_values: ReferenceValues | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Wrapper for :class:`ColdDeckImputer`."""
    return ColdDeckImputer(
        reference_values=reference_values, random_state=random_state
    ).impute(df)


def hybrid_impute(
    df: pd.DataFrame, methods: list[BaseImputer] | None = None
) -> pd.DataFrame:
    """Wrapper for :class:`HybridImputer`."""
    return HybridImputer(methods=methods).impute(df)


def bayesian_ridge_impute(
    df: pd.DataFrame, max_iter: int = 300, tol: float = 1e-3
) -> pd.DataFrame:
    """Wrapper for :class:`BayesianRidgeImputer`."""
    return BayesianRidgeImputer(max_iter=max_iter, tol=tol).impute(df)


def stacking_impute(
    df: pd.DataFrame,
    base_imputers: list[BaseImputer] | None = None,
    meta_strategy: str = "mean",
) -> pd.DataFrame:
    """Wrapper for :class:`StackingImputer`."""
    return StackingImputer(
        base_imputers=base_imputers, meta_strategy=meta_strategy
    ).impute(df)


def bagging_impute(
    df: pd.DataFrame,
    base_imputer: BaseImputer | None = None,
    n_estimators: int = 10,
    max_samples: float = 0.8,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Wrapper for :class:`BaggingImputer`."""
    return BaggingImputer(
        base_imputer=base_imputer,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
    ).impute(df)


def radius_neighbors_impute(
    df: pd.DataFrame, radius: float = 1.0, weights: str = "distance"
) -> pd.DataFrame:
    """Wrapper for :class:`RadiusNeighborsImputer`."""
    return RadiusNeighborsImputer(radius=radius, weights=weights).impute(df)


def local_mean_impute(
    df: pd.DataFrame, n_neighbors: int = 5, distance_weight_power: float = 2.0
) -> pd.DataFrame:
    """Wrapper for :class:`LocalMeanImputer`."""
    return LocalMeanImputer(
        n_neighbors=n_neighbors, distance_weight_power=distance_weight_power
    ).impute(df)


def huber_impute(
    df: pd.DataFrame, epsilon: float = 1.35, max_iter: int = 100
) -> pd.DataFrame:
    """Wrapper for :class:`HuberImputer`."""
    return HuberImputer(epsilon=epsilon, max_iter=max_iter).impute(df)


def ransac_impute(
    df: pd.DataFrame, max_trials: int = 100, random_state: int | None = None
) -> pd.DataFrame:
    """Wrapper for :class:`RANSACImputer`."""
    return RANSACImputer(max_trials=max_trials, random_state=random_state).impute(df)


def trimmed_mean_impute(df: pd.DataFrame, trim_fraction: float = 0.1) -> pd.DataFrame:
    """Wrapper for :class:`TrimmedMeanImputer`."""
    return TrimmedMeanImputer(trim_fraction=trim_fraction).impute(df)
