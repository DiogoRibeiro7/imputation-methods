"""A unified pandas API for missing-data imputation.

Every imputer subclasses :class:`~imputation_methods.base.BaseImputer` and exposes
a single ``impute(df)`` method that takes a numeric :class:`pandas.DataFrame`
and returns a new, imputed one. Each imputer also has a functional shortcut,
e.g. ``mean_impute(df)`` for ``MeanImputer().impute(df)``.

Examples:
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imputation_methods import MedianImputer
    >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 10.0]})
    >>> MedianImputer().impute(df)["a"].tolist()
    [1.0, 3.0, 3.0, 10.0]
"""

from importlib.metadata import PackageNotFoundError, version

from .base import BaseImputer
from .ensemble import BaggingImputer, HybridImputer, StackingImputer
from .functional import (
    autoencoder_impute,
    bagging_impute,
    bayesian_pca_impute,
    bayesian_ridge_impute,
    cold_deck_impute,
    constant_impute,
    em_impute,
    end_of_distribution_impute,
    forward_fill_fallback_impute,
    gain_impute,
    gaussian_process_impute,
    group_mean_impute,
    hot_deck_impute,
    huber_impute,
    hybrid_impute,
    indicator_impute,
    interpolation_impute,
    kalman_filter_impute,
    knn_impute,
    linear_trend_impute,
    local_mean_impute,
    locf_impute,
    mean_impute,
    median_impute,
    mice_impute,
    miss_forest_impute,
    mode_impute,
    moving_average_impute,
    nocb_impute,
    polynomial_trend_impute,
    predictive_mean_matching,
    quantile_impute,
    radius_neighbors_impute,
    random_sampling_impute,
    ransac_impute,
    regression_impute,
    seasonal_impute,
    soft_impute,
    stacking_impute,
    stochastic_regression_impute,
    trimmed_mean_impute,
    weighted_moving_average_impute,
)
from .iterative import EMImputer, GAINImputer, MICEImputer, MissForestImputer
from .matrix import BayesianPCAImputer, SoftImputeImputer
from .metrics import mae, rmse
from .neighbors import KNNImputerMethod, LocalMeanImputer, RadiusNeighborsImputer
from .neural import AutoencoderImputer
from .regression import (
    BayesianRidgeImputer,
    GaussianProcessImputer,
    HuberImputer,
    PMMImputer,
    RANSACImputer,
    RegressionImputer,
    StochasticRegressionImputer,
)
from .sampling import ColdDeckImputer, HotDeckImputer, RandomSamplingImputer
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

try:
    __version__ = version("imputation-methods")
except PackageNotFoundError:  # pragma: no cover - package is not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "AutoencoderImputer",
    "BaggingImputer",
    "BaseImputer",
    "BayesianPCAImputer",
    "BayesianRidgeImputer",
    "ColdDeckImputer",
    "ConstantImputer",
    "EMImputer",
    "EndOfDistributionImputer",
    "ForwardFillFallbackImputer",
    "GAINImputer",
    "GaussianProcessImputer",
    "GroupMeanImputer",
    "HotDeckImputer",
    "HuberImputer",
    "HybridImputer",
    "IndicatorImputer",
    "InterpolationImputer",
    "KNNImputerMethod",
    "KalmanFilterImputer",
    "LOCFImputer",
    "LinearTrendImputer",
    "LocalMeanImputer",
    "MICEImputer",
    "MeanImputer",
    "MedianImputer",
    "MissForestImputer",
    "ModeImputer",
    "MovingAverageImputer",
    "NOCBImputer",
    "PMMImputer",
    "PolynomialTrendImputer",
    "QuantileImputer",
    "RANSACImputer",
    "RadiusNeighborsImputer",
    "RandomSamplingImputer",
    "RegressionImputer",
    "SeasonalImputer",
    "SoftImputeImputer",
    "StackingImputer",
    "StochasticRegressionImputer",
    "TrimmedMeanImputer",
    "WeightedMovingAverageImputer",
    "__version__",
    "autoencoder_impute",
    "bagging_impute",
    "bayesian_pca_impute",
    "bayesian_ridge_impute",
    "cold_deck_impute",
    "constant_impute",
    "em_impute",
    "end_of_distribution_impute",
    "forward_fill_fallback_impute",
    "gain_impute",
    "gaussian_process_impute",
    "group_mean_impute",
    "hot_deck_impute",
    "huber_impute",
    "hybrid_impute",
    "indicator_impute",
    "interpolation_impute",
    "kalman_filter_impute",
    "knn_impute",
    "linear_trend_impute",
    "local_mean_impute",
    "locf_impute",
    "mae",
    "mean_impute",
    "median_impute",
    "mice_impute",
    "miss_forest_impute",
    "mode_impute",
    "moving_average_impute",
    "nocb_impute",
    "polynomial_trend_impute",
    "predictive_mean_matching",
    "quantile_impute",
    "radius_neighbors_impute",
    "random_sampling_impute",
    "ransac_impute",
    "regression_impute",
    "rmse",
    "seasonal_impute",
    "soft_impute",
    "stacking_impute",
    "stochastic_regression_impute",
    "trimmed_mean_impute",
    "weighted_moving_average_impute",
]
