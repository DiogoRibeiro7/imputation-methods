# Imputation Showcase

This repository demonstrates various techniques for handling missing data.

## Available Methods

- Mean imputation
- Median imputation
- K-Nearest Neighbors (KNN)
- Predictive Mean Matching (PMM)
- Multiple Imputation by Chained Equations (MICE)
- Regression imputation
- Stochastic regression imputation
- Last Observation Carried Forward (LOCF)
- Next Observation Carried Backward (NOCB)

## Usage

The imputation classes are implemented in `src/imputation_methods.py` and can be imported directly.

```python
from src.imputation_methods import (
    MeanImputer,
    MedianImputer,
    KNNImputerMethod,
    PMMImputer,
    MICEImputer,
    RegressionImputer,
    StochasticRegressionImputer,
    LOCFImputer,
    NOCBImputer,
)

df = pd.read_csv("data/example.csv")
df_imputed = MeanImputer().impute(df)
```

The imputers expect all columns to be numeric. Provide only numerical data or
encode categorical features before calling the imputation routines.

## Evaluation Metrics

The module also provides `rmse` and `mae` helpers to evaluate imputation quality.

## Continuous Integration

Linting and unit tests run automatically via GitHub Actions on each push.

An example dataset is available in the `data/` directory for quick experimentation.

Tests can be executed with `pytest`.
