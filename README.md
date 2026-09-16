# imputation-methods

[![PyPI](https://img.shields.io/pypi/v/imputation-methods)](https://pypi.org/project/imputation-methods/)
[![Python versions](https://img.shields.io/pypi/pyversions/imputation-methods)](https://pypi.org/project/imputation-methods/)
[![CI](https://github.com/DiogoRibeiro7/imputation-methods/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/imputation-methods/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/DiogoRibeiro7/imputation-methods/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**42 missing-data imputation methods behind one pandas API.** Swap mean imputation
for KNN, MICE, a Kalman filter or low-rank matrix completion by changing one line,
and compare them with the same evaluation code.

```python
from imputation_methods import KNNImputerMethod

completed = KNNImputerMethod(k=5).impute(df)
```

- **One interface.** Every imputer takes a numeric `DataFrame` and returns a new one
  with the same index and columns. The input is never modified.
- **Broad coverage.** Statistical, donor-based, time-series, nearest-neighbor,
  regression, iterative, matrix-completion, neural and ensemble methods.
- **Light dependencies.** NumPy, pandas, SciPy and scikit-learn only.
- **Typed and tested.** Inline type hints checked by mypy in strict mode, and tests
  on Python 3.10–3.14, on the newest and the oldest supported dependency versions.

## Installation

```bash
pip install imputation-methods
```

The optional `viz` extra installs matplotlib and seaborn, used by the example
notebooks and scripts:

```bash
pip install "imputation-methods[viz]"
```

Requires Python 3.10 or newer.

## Quick start

```python
import numpy as np
import pandas as pd

from imputation_methods import KNNImputerMethod, MeanImputer, knn_impute

df = pd.DataFrame(
    {
        "height": [170.0, 165.0, np.nan, 180.0, 175.0],
        "weight": [65.0, np.nan, 70.0, 85.0, 78.0],
        "age": [30.0, 25.0, 35.0, np.nan, 40.0],
    }
)

mean_filled = MeanImputer().impute(df)
knn_filled = KNNImputerMethod(k=2).impute(df)

# Every imputer also has a functional shortcut.
same_as_knn = knn_impute(df, k=2)
```

Imputers are configured in the constructor. Those with a random component accept
`random_state` for reproducible results.

## Available methods

| Family | Imputers |
| --- | --- |
| Statistical | `MeanImputer`, `MedianImputer`, `ModeImputer`, `ConstantImputer`, `QuantileImputer`, `TrimmedMeanImputer`, `EndOfDistributionImputer`, `GroupMeanImputer`, `IndicatorImputer` |
| Donor sampling | `RandomSamplingImputer`, `HotDeckImputer`, `ColdDeckImputer` |
| Time series | `LOCFImputer`, `NOCBImputer`, `ForwardFillFallbackImputer`, `InterpolationImputer`, `MovingAverageImputer`, `WeightedMovingAverageImputer`, `LinearTrendImputer`, `PolynomialTrendImputer`, `SeasonalImputer`, `KalmanFilterImputer` |
| Nearest neighbors | `KNNImputerMethod`, `RadiusNeighborsImputer`, `LocalMeanImputer` |
| Regression | `RegressionImputer`, `StochasticRegressionImputer`, `PMMImputer` (predictive mean matching), `BayesianRidgeImputer`, `HuberImputer`, `RANSACImputer`, `GaussianProcessImputer` |
| Iterative | `MICEImputer`, `MissForestImputer`, `EMImputer` |
| Matrix completion | `SoftImputeImputer`, `BayesianPCAImputer` (probabilistic PCA) |
| Neural networks | `AutoencoderImputer`, `GAINImputer` (generative adversarial imputation) |
| Ensembles | `HybridImputer` (fallback chain), `StackingImputer`, `BaggingImputer` (bootstrap aggregating) |

`EMImputer` runs iterative chained-equation imputation rather than closed-form EM
for a multivariate normal distribution.

The [API reference](https://diogoribeiro7.github.io/imputation-methods/api/) documents
every class and its parameters.

## Evaluating an imputation

When you have complete data, hide some values, impute them, and score only the
cells you hid:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

from imputation_methods import KNNImputerMethod, MeanImputer, MICEImputer, mae, rmse

complete = load_diabetes(as_frame=True).data
rng = np.random.default_rng(0)
mask = rng.random(complete.shape) < 0.2
incomplete = complete.mask(mask)

imputers = {
    "mean": MeanImputer(),
    "knn": KNNImputerMethod(k=5),
    "mice": MICEImputer(random_state=0),
}
for name, imputer in imputers.items():
    completed = imputer.impute(incomplete)
    true = pd.Series(complete.to_numpy()[mask])
    pred = pd.Series(completed.to_numpy()[mask])
    print(f"{name:>5}: RMSE={rmse(true, pred):.4f}  MAE={mae(true, pred):.4f}")
```

## Input requirements

- A `pandas.DataFrame` with numeric columns; missing values as `NaN` (or `pd.NA` in
  nullable dtypes). Encode categorical columns before imputing. `GroupMeanImputer`
  is the exception: its grouping column may be non-numeric.
- Time-series imputers use row order, so sort the data first.
- Columns with no observed values are left as `NaN` by most imputers.

## Documentation

Full documentation, including guides on choosing a method and evaluating results:
<https://diogoribeiro7.github.io/imputation-methods/>

The repository also has [example scripts](https://github.com/DiogoRibeiro7/imputation-methods/tree/main/examples)
and [Jupyter notebooks](https://github.com/DiogoRibeiro7/imputation-methods/tree/main/notebooks).

## Contributing

Contributions are welcome. See the
[contributing guide](https://github.com/DiogoRibeiro7/imputation-methods/blob/main/CONTRIBUTING.md)
for the development setup, and the
[code of conduct](https://github.com/DiogoRibeiro7/imputation-methods/blob/main/CODE_OF_CONDUCT.md).
Report security issues as described in the
[security policy](https://github.com/DiogoRibeiro7/imputation-methods/blob/main/SECURITY.md).

## Citation

If you use this library in research, please cite it. Citation metadata is in
[`CITATION.cff`](https://github.com/DiogoRibeiro7/imputation-methods/blob/main/CITATION.cff);
GitHub's "Cite this repository" button exports it as BibTeX or APA.

## License

MIT. See [LICENSE](https://github.com/DiogoRibeiro7/imputation-methods/blob/main/LICENSE).
