# Imputation Showcase

This repository demonstrates various techniques for handling missing data.

## Available Methods

- Mean imputation
- Median imputation
- K-Nearest Neighbors (KNN)
- Predictive Mean Matching (PMM)
- Multiple Imputation by Chained Equations (MICE)

## Usage

The imputation classes are implemented in `src/imputation_methods.py` and can be imported directly.

```python
from src.imputation_methods import (
    MeanImputer,
    MedianImputer,
    KNNImputerMethod,
    PMMImputer,
    MICEImputer,
)

df = pd.read_csv("data/example.csv")
df_imputed = MeanImputer().impute(df)
```

The imputers expect all columns to be numeric. Provide only numerical data or
encode categorical features before calling the imputation routines.

Tests can be executed with `pytest`.
