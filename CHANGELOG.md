# Changelog

<!-- --8<-- [start:changelog] -->

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First release on PyPI, as `imputation-methods` 0.1.0.

### Added

- 42 imputers behind one interface, `BaseImputer.impute(df) -> DataFrame`:
  - **Statistical**: `MeanImputer`, `MedianImputer`, `ModeImputer`, `ConstantImputer`,
    `QuantileImputer`, `TrimmedMeanImputer`, `EndOfDistributionImputer`,
    `GroupMeanImputer`, `IndicatorImputer`
  - **Donor sampling**: `RandomSamplingImputer`, `HotDeckImputer`, `ColdDeckImputer`
  - **Time series**: `LOCFImputer`, `NOCBImputer`, `ForwardFillFallbackImputer`,
    `InterpolationImputer`, `MovingAverageImputer`, `WeightedMovingAverageImputer`,
    `LinearTrendImputer`, `PolynomialTrendImputer`, `SeasonalImputer`,
    `KalmanFilterImputer`
  - **Nearest neighbors**: `KNNImputerMethod`, `RadiusNeighborsImputer`,
    `LocalMeanImputer`
  - **Regression**: `RegressionImputer`, `StochasticRegressionImputer`, `PMMImputer`,
    `BayesianRidgeImputer`, `HuberImputer`, `RANSACImputer`, `GaussianProcessImputer`
  - **Iterative**: `MICEImputer`, `EMImputer`, `MissForestImputer`
  - **Matrix completion**: `SoftImputeImputer`, `BayesianPCAImputer`
  - **Neural networks**: `AutoencoderImputer`, `GAINImputer`
  - **Ensembles**: `HybridImputer`, `StackingImputer`, `BaggingImputer`
- A functional shortcut for every imputer, e.g. `knn_impute(df, k=3)`.
- `rmse` and `mae` metrics for scoring imputations against ground truth.
- Inline type hints (`py.typed`), checked with mypy in strict mode.
- Documentation site with an API reference generated from docstrings.

### Changed

- Renamed the project from `imputation-showcase` to `imputation-methods` and moved
  the code to a `src/` layout split into submodules. Import everything from the
  top-level package: `from imputation_methods import MeanImputer`.
- `SoftImputeImputer` and `BayesianPCAImputer` are now implemented directly on NumPy,
  replacing the unmaintained `fancyimpute` and `ppca` dependencies. `fancyimpute`
  0.7.0 no longer worked with scikit-learn 1.8+, and it pulled `pytest`, `nose`,
  `cvxpy` and `cvxopt` in as runtime dependencies. Both imputers are now
  deterministic and gained validated `shrinkage_value`/`convergence_threshold` and
  `max_iter`/`tol` options respectively.
- Runtime dependencies are now just NumPy (>=1.24), pandas (>=2.0, including 3.x),
  SciPy (>=1.10) and scikit-learn (>=1.4). Matplotlib and seaborn moved to the
  optional `viz` extra.
- Handled fallbacks (e.g. MICE falling back to mean imputation) log a single
  `WARNING` instead of `ERROR` + `WARNING`, and routine progress messages were
  removed from `INFO`.
- `GAINImputer` now implements Generative Adversarial Imputation Nets (Yoon et
  al., 2018) on NumPy, with the reference hyperparameters (`batch_size`,
  `hint_rate`, `alpha`, `iterations`, `learning_rate`). It previously ran
  scikit-learn's `IterativeImputer` and gave the same results as `MICEImputer`.
- `BaggingImputer` now performs real bootstrap aggregating: each run imputes a
  resampled set of rows (`max_samples`) with its own seed, and the results are
  averaged. It previously averaged identical runs on the full data.
- `ColdDeckImputer` accepts `random_state` for reproducible sampling from array
  reference values.
- Faster, vectorized `GroupMeanImputer` and `SeasonalImputer`.

### Fixed

- `StochasticRegressionImputer` raised `ValueError` whenever more than one column had
  missing values; `RegressionImputer`, `PMMImputer` and `GaussianProcessImputer`
  silently fell back to mean imputation in the same situation. Gaps in predictor
  columns are now mean-filled before fitting.
- A column with no observed values made `EMImputer` raise, and made `MICEImputer`,
  `MissForestImputer` and `KNNImputerMethod` fall back to mean/median
  imputation for every column. Such columns are now left as `NaN` and the other
  columns are imputed normally.
- `PMMImputer` drew donors with the same seed for every missing value, and filled
  columns with no observed values with 0.
- Time-series imputers used `fillna(method=...)`, which was removed in pandas 3.
- `HotDeckImputer` triggered a pandas 4 deprecation warning when stratifying by a
  single column.
- Docstring examples that used non-existent arguments (e.g. `n_neighbors=`).

<!-- --8<-- [end:changelog] -->

[Unreleased]: https://github.com/DiogoRibeiro7/imputation-methods/commits/main
