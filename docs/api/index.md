# API reference

Everything listed here can be imported directly from the top-level package:

```python
from imputation_methods import MeanImputer, KNNImputerMethod, rmse
```

The submodules below only group related imputers; you don't need them for imports.

## The imputer interface

Every imputer subclasses [`BaseImputer`](base.md) and exposes one method,
`impute(df) -> DataFrame`:

- **Input**: a `pandas.DataFrame` with numeric columns, where missing values are
  `NaN` (or `pd.NA` in nullable dtypes). Non-numeric columns raise `TypeError`.
  `GroupMeanImputer` is the exception: its grouping column may be non-numeric.
- **Output**: a new dataframe with the same index and columns. The input is never
  modified. `IndicatorImputer` also appends one indicator column per input column.
- **Configuration** happens in the constructor. Imputers with randomness accept a
  `random_state` for reproducible results.
- **Columns with no observed values** carry no information, so most imputers leave
  them as `NaN`. `ConstantImputer` fills them with its constant, and
  `HybridImputer`, `BaggingImputer`, `BayesianRidgeImputer`, `HuberImputer`,
  `LocalMeanImputer` and `AutoencoderImputer` fall back to 0 (or a value
  reconstructed from 0). Drop or handle empty columns explicitly if that matters.
- **Row order** matters only for the [time-series imputers](time-series.md); sort
  your data first.

Each imputer also has a [functional shortcut](functional.md), for example
`knn_impute(df, k=3)` for `KNNImputerMethod(k=3).impute(df)`.

## Modules

| Page | Module | Contents |
| --- | --- | --- |
| [Base class](base.md) | `imputation_methods.base` | `BaseImputer` |
| [Statistical](statistical.md) | `imputation_methods.statistical` | `MeanImputer`, `MedianImputer`, `ModeImputer`, `ConstantImputer`, `QuantileImputer`, `TrimmedMeanImputer`, `EndOfDistributionImputer`, `GroupMeanImputer`, `IndicatorImputer` |
| [Donor sampling](sampling.md) | `imputation_methods.sampling` | `RandomSamplingImputer`, `HotDeckImputer`, `ColdDeckImputer` |
| [Time series](time-series.md) | `imputation_methods.time_series` | `LOCFImputer`, `NOCBImputer`, `ForwardFillFallbackImputer`, `InterpolationImputer`, `MovingAverageImputer`, `WeightedMovingAverageImputer`, `LinearTrendImputer`, `PolynomialTrendImputer`, `SeasonalImputer`, `KalmanFilterImputer` |
| [Nearest neighbors](neighbors.md) | `imputation_methods.neighbors` | `KNNImputerMethod`, `RadiusNeighborsImputer`, `LocalMeanImputer` |
| [Regression](regression.md) | `imputation_methods.regression` | `RegressionImputer`, `StochasticRegressionImputer`, `PMMImputer`, `BayesianRidgeImputer`, `HuberImputer`, `RANSACImputer`, `GaussianProcessImputer` |
| [Iterative](iterative.md) | `imputation_methods.iterative` | `MICEImputer`, `EMImputer`, `MissForestImputer`, `GAINImputer` |
| [Matrix completion](matrix.md) | `imputation_methods.matrix` | `SoftImputeImputer`, `BayesianPCAImputer` |
| [Neural networks](neural.md) | `imputation_methods.neural` | `AutoencoderImputer` |
| [Ensembles](ensemble.md) | `imputation_methods.ensemble` | `HybridImputer`, `StackingImputer`, `BaggingImputer` |
| [Functional API](functional.md) | `imputation_methods.functional` | `mean_impute`, `median_impute`, … (42 functions) |
| [Metrics](metrics.md) | `imputation_methods.metrics` | `rmse`, `mae` |
