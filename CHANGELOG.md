# Changelog

<!-- --8<-- [start:changelog] -->

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-16

A consistent API: one name per concept, an explicit choice when a model fails, and
one policy for empty columns and output dtypes across all 42 imputers. Old names
keep working, with a `FutureWarning`, until 1.0.0.

### Added

- `on_error` parameter on the imputers whose model can fail to fit (`MICEImputer`,
  `MissForestImputer`, `KNNImputer`, `RadiusNeighborsImputer`, `RegressionImputer`,
  `PMMImputer`, `GaussianProcessImputer`, `RANSACImputer`, `SoftImputeImputer`,
  `PPCAImputer`, `AutoencoderImputer`) and on their shortcuts:
  - `on_error="raise"` raises the new `ImputationError` (a `RuntimeError`), chained
    to the underlying error.
  - `on_error="fallback"` uses mean or median imputation instead, as before, and
    logs a warning.

### Changed

- Consistent parameter names across imputers, following scikit-learn conventions:
  - `max_iter` for iteration budgets: `SoftImputeImputer(max_iters=...)` and
    `GAINImputer(iterations=...)` are now `max_iter=...`.
  - `n_neighbors` for neighbor and donor counts: `KNNImputer(k=...)` and
    `PMMImputer(k=...)` are now `n_neighbors=...`.
  - `n_std` for the number of standard deviations in
    `EndOfDistributionImputer(k=...)`.
  - `strategy` for the summary statistic in `GroupMeanImputer`,
    `MovingAverageImputer` and `SeasonalImputer`, which used `method=...`.
    `InterpolationImputer(method=...)` is unchanged; there it selects the
    interpolation kind, as in pandas.
  - The functional shortcuts follow the same names.
- Renamed `KNNImputerMethod` to `KNNImputer`, `BayesianPCAImputer` to `PPCAImputer`
  (it is maximum-likelihood probabilistic PCA), `predictive_mean_matching` to
  `pmm_impute` and `bayesian_pca_impute` to `ppca_impute`.
- Instance attributes use the new parameter names (e.g. `imputer.n_neighbors`).
- **One output dtype policy for every imputer.** Columns without missing values are
  returned unchanged, with the same dtype; imputed columns are floating point,
  keeping `float32` or `float64`, and pandas nullable columns become `Float64` so
  values that couldn't be imputed stay `<NA>`. Previously the imputers based on
  scikit-learn (`KNNImputer`, `MICEImputer`, `MissForestImputer`, `EMImputer`,
  `SoftImputeImputer`, `PPCAImputer`, `GAINImputer`) turned every column, including
  complete integer columns, into `float64`.
- **Behavior change:** a column with no observed values is now left as `NaN` by
  every imputer that learns from the data. `BayesianRidgeImputer`, `HuberImputer`,
  `LocalMeanImputer` and `HybridImputer` used to fill it with 0, and
  `AutoencoderImputer` with values reconstructed from 0. `HybridImputer` therefore
  no longer guarantees a result without `NaN` when a column is entirely empty; add
  `ConstantImputer` to its `methods` to fill such columns with a fixed value.

### Deprecated

- The old class, function and parameter names above still work but emit a
  `FutureWarning` starting with `imputation-methods:`, and will be removed in 1.0.0.
  Run your code with `-W "error:imputation-methods:FutureWarning"` to find uses.
- Falling back to a simpler method without asking. When a model fails and `on_error`
  isn't set, the imputer still falls back but now emits a `FutureWarning`; the
  default will become `on_error="raise"` in 1.0.0. Unexpected errors that previously
  raised a plain `RuntimeError` now raise `ImputationError`, a subclass.

### Fixed

- 24 imputers raised `TypeError` on pandas nullable integer columns (`Int64`) with
  missing values, and `KalmanFilterImputer` and `LocalMeanImputer` on other nullable
  columns; `float32` columns with missing values failed in 7 imputers under pandas 3.
- `TrimmedMeanImputer` emitted a SciPy `SmallSampleWarning`, and `MedianImputer`,
  `ModeImputer`, `IndicatorImputer`, `GroupMeanImputer` and `ColdDeckImputer` a NumPy
  "Mean of empty slice" warning on older NumPy, for columns with no observed values.

## [0.1.0] - 2026-09-16

First release on PyPI, as `imputation-methods`.

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
- A functional shortcut for every imputer, e.g. `knn_impute(df, k=3)`, accepting
  the same parameters and defaults as its class.
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
- `AutoencoderImputer` triggered a scikit-learn `DataConversionWarning` on
  single-column input.

[Unreleased]: https://github.com/DiogoRibeiro7/imputation-methods/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/DiogoRibeiro7/imputation-methods/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DiogoRibeiro7/imputation-methods/releases/tag/v0.1.0

<!-- --8<-- [end:changelog] -->
