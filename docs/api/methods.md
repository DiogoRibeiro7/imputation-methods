# API Reference: Imputation Methods

Complete API documentation for all imputation methods.

## Base Class

### `BaseImputer`

Abstract base class for all imputation methods.

```python
from imputation_showcase import BaseImputer
```

**Methods:**

#### `impute(df: pd.DataFrame) -> pd.DataFrame`

Impute missing values in the DataFrame.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with missing values (NaN)

**Returns:**
- pd.DataFrame: DataFrame with imputed values

**Raises:**
- TypeError: If DataFrame contains non-numeric columns

**Protected Methods:**

#### `_ensure_numeric(df: pd.DataFrame) -> pd.DataFrame`

Validate that DataFrame contains only numeric columns.

---

## Statistical Methods

### `MeanImputer`

Replace missing values with column means.

```python
from imputation_showcase import MeanImputer

imputer = MeanImputer()
df_imputed = imputer.impute(df)
```

**Methods:**
- `impute(df)`: Returns DataFrame with NaN replaced by column means

---

### `MedianImputer`

Replace missing values with column medians.

```python
from imputation_showcase import MedianImputer

imputer = MedianImputer()
df_imputed = imputer.impute(df)
```

**Methods:**
- `impute(df)`: Returns DataFrame with NaN replaced by column medians

---

## Time Series Methods

### `LOCFImputer`

Last Observation Carried Forward imputation.

```python
from imputation_showcase import LOCFImputer

imputer = LOCFImputer()
df_imputed = imputer.impute(df)
```

**Methods:**
- `impute(df)`: Forward-fill missing values using pandas `ffill()`

---

### `NOCBImputer`

Next Observation Carried Backward imputation.

```python
from imputation_showcase import NOCBImputer

imputer = NOCBImputer()
df_imputed = imputer.impute(df)
```

**Methods:**
- `impute(df)`: Backward-fill missing values using pandas `bfill()`

---

## Distance-Based Methods

### `KNNImputerMethod`

K-Nearest Neighbors imputation.

```python
from imputation_showcase import KNNImputerMethod

imputer = KNNImputerMethod(k=5)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `k` (int): Number of neighbors to consider. Default: 5

**Methods:**
- `impute(df)`: Impute using weighted average of k-nearest neighbors

**Raises:**
- TypeError: If k is not an integer
- ValueError: If k is not positive

---

### `HotDeckImputer`

Hot Deck imputation by random sampling from donors.

```python
from imputation_showcase import HotDeckImputer

imputer = HotDeckImputer(
    stratify_cols=['category'],
    random_state=42
)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `stratify_cols` (list[str] | None): Columns to stratify by. Default: None
- `random_state` (int | None): Random seed. Default: None

**Methods:**
- `impute(df)`: Randomly sample donors, optionally stratified by specified columns

---

## Regression-Based Methods

### `RegressionImputer`

Linear regression-based imputation.

```python
from imputation_showcase import RegressionImputer

imputer = RegressionImputer()
df_imputed = imputer.impute(df)
```

**Methods:**
- `impute(df)`: Predict missing values using linear regression on other columns

---

### `StochasticRegressionImputer`

Stochastic regression imputation (regression + noise).

```python
from imputation_showcase import StochasticRegressionImputer

imputer = StochasticRegressionImputer(random_state=42)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `random_state` (int | None): Random seed for noise generation. Default: None

**Methods:**
- `impute(df)`: Predict using regression and add Gaussian noise to preserve variance

---

### `PMMImputer`

Predictive Mean Matching imputation.

```python
from imputation_showcase import PMMImputer

imputer = PMMImputer(k=5, random_state=42)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `k` (int): Number of donor candidates. Default: 5
- `random_state` (int | None): Random seed. Default: None

**Methods:**
- `impute(df)`: Predict using regression, then select actual observed value from k nearest predictions

**Raises:**
- TypeError: If k is not an integer
- ValueError: If k is not positive

---

### `MICEImputer`

Multiple Imputation by Chained Equations.

```python
from imputation_showcase import MICEImputer

imputer = MICEImputer(random_state=42)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `random_state` (int | None): Random seed. Default: None

**Methods:**
- `impute(df)`: Iteratively impute each variable using the others as predictors

**Note:** This implementation returns a single imputation. For proper multiple imputation with uncertainty estimates, consider specialized packages.

---

## Tree-Based Methods

### `MissForestImputer`

Random Forest-based iterative imputation.

```python
from imputation_showcase import MissForestImputer

imputer = MissForestImputer(random_state=42)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `random_state` (int | None): Random seed. Default: None

**Methods:**
- `impute(df)`: Iteratively impute using Random Forest regressor

---

## Matrix Completion Methods

### `SoftImputeImputer`

Matrix completion via nuclear-norm regularization.

```python
from imputation_showcase import SoftImputeImputer

imputer = SoftImputeImputer(
    max_iters=100,
    init_fill_method='zero'
)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `max_iters` (int): Maximum iterations. Default: 100
- `init_fill_method` (str): Initialization method ('zero', 'mean'). Default: 'zero'

**Methods:**
- `impute(df)`: Low-rank matrix completion using SoftImpute algorithm

---

### `BayesianPCAImputer`

Probabilistic PCA-based imputation.

```python
from imputation_showcase import BayesianPCAImputer

imputer = BayesianPCAImputer(
    n_components=2,
    min_obs=1
)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `n_components` (int | None): Number of latent dimensions. Default: 1
- `min_obs` (int): Minimum observations required per column. Default: 1

**Methods:**
- `impute(df)`: Impute using probabilistic PCA

**Note:** Falls back to mean imputation if PPCA fails.

---

## Deep Learning Methods

### `AutoencoderImputer`

Autoencoder-based imputation.

```python
from imputation_showcase import AutoencoderImputer

imputer = AutoencoderImputer(
    hidden_layer_sizes=(10, 5),
    max_iter=200,
    random_state=42
)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `hidden_layer_sizes` (tuple[int, ...]): Network architecture. Default: (10,)
- `max_iter` (int): Maximum training iterations. Default: 200
- `random_state` (int | None): Random seed. Default: None

**Methods:**
- `impute(df)`: Impute using trained autoencoder reconstruction

---

### `GAINImputer`

Generative Adversarial Imputation Networks (simplified).

```python
from imputation_showcase import GAINImputer

imputer = GAINImputer(random_state=42)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `random_state` (int | None): Random seed. Default: None

**Methods:**
- `impute(df)`: Iterative imputation approximating GAIN

**Note:** This is a simplified implementation using IterativeImputer.

---

## Advanced Statistical Methods

### `GaussianProcessImputer`

Gaussian Process regression-based imputation.

```python
from imputation_showcase import GaussianProcessImputer
from sklearn.gaussian_process.kernels import RBF

imputer = GaussianProcessImputer(
    kernel=RBF(length_scale=1.0),
    alpha=1e-10,
    random_state=42
)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `kernel` (RBF | None): Covariance kernel. Default: RBF()
- `alpha` (float): Noise level for numerical stability. Default: 1e-10
- `random_state` (int | None): Random seed. Default: None

**Methods:**
- `impute(df)`: Predict missing values using Gaussian Process regression

---

## Functional API

For convenience, functional wrappers are provided:

```python
from imputation_showcase import (
    mean_impute,
    median_impute,
    knn_impute,
    predictive_mean_matching,
    mice_impute,
    regression_impute,
    stochastic_regression_impute,
    locf_impute,
    nocb_impute,
    hot_deck_impute,
    miss_forest_impute,
    bayesian_pca_impute,
    soft_impute,
    autoencoder_impute,
    gain_impute,
    gaussian_process_impute
)

# Example usage
df_imputed = knn_impute(df, k=5)
```

These functions create an imputer instance and call `impute()` in one step.

---

## Type Hints

All methods use proper type hints:

```python
from typing import Union
import pandas as pd
import numpy as np

def impute(self, df: pd.DataFrame) -> pd.DataFrame:
    """Type-checked imputation."""
    ...
```

---

## Common Parameters

Many methods share common parameters:

- `random_state` (int | None): Random seed for reproducibility
- `k` (int): Number of neighbors/donors for neighbor-based methods
- `max_iter` (int): Maximum iterations for iterative methods

---

## Error Handling

All methods validate input and raise appropriate errors:

```python
try:
    df_imputed = imputer.impute(df)
except TypeError as e:
    # Non-numeric columns present
    print(f"Type error: {e}")
except ValueError as e:
    # Invalid parameter values
    print(f"Value error: {e}")
```

---

## Performance Characteristics

| Method | Time Complexity | Space Complexity | Scalability |
|--------|-----------------|------------------|-------------|
| Mean/Median | O(n) | O(1) | Excellent |
| LOCF/NOCB | O(n) | O(1) | Excellent |
| KNN | O(n²) | O(n) | Moderate |
| Regression | O(n·p²) | O(p²) | Good |
| MICE | O(i·n·p²) | O(p²) | Moderate |
| MissForest | O(i·t·n·log(n)) | O(n) | Poor |
| GP | O(n³) | O(n²) | Poor |

Where:
- n = number of rows
- p = number of columns
- i = number of iterations
- t = number of trees

---

## See Also

- [Evaluation Metrics API](evaluation.md)
- [User Guide](../user-guide/methods.md) for detailed method descriptions
- [Examples](../examples/time-series.md) for usage patterns
