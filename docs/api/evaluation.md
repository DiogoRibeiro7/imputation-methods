# API Reference: Evaluation Metrics

Complete API documentation for evaluation functions.

## Evaluation Functions

### `rmse`

Calculate Root Mean Squared Error between true and imputed values.

```python
from imputation_showcase import rmse

error = rmse(y_true, y_pred)
```

**Signature:**
```python
def rmse(true: pd.Series, pred: pd.Series) -> float
```

**Parameters:**
- `true` (pd.Series): Ground truth values
- `pred` (pd.Series): Predicted or imputed values

**Returns:**
- `float`: Root mean squared error

**Formula:**

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

**Example:**
```python
import pandas as pd
import numpy as np
from imputation_showcase import KNNImputerMethod, rmse

# Original complete data
df_complete = pd.DataFrame({'A': [1, 2, 3, 4, 5]})

# Introduce missingness
df_missing = df_complete.copy()
df_missing.loc[[1, 3], 'A'] = np.nan

# Impute
imputer = KNNImputerMethod(k=2)
df_imputed = imputer.impute(df_missing)

# Evaluate (on originally missing values only)
missing_mask = df_missing['A'].isna()
error = rmse(
    df_complete.loc[missing_mask, 'A'],
    df_imputed.loc[missing_mask, 'A']
)

print(f"RMSE: {error:.4f}")
```

**Use Cases:**
- Comparing imputation methods
- Evaluating imputation quality when ground truth is available
- Tuning hyperparameters

**Notes:**
- Units are same as original data
- More sensitive to outliers than MAE
- Larger errors are penalized more heavily

---

### `mae`

Calculate Mean Absolute Error between true and imputed values.

```python
from imputation_showcase import mae

error = mae(y_true, y_pred)
```

**Signature:**
```python
def mae(true: pd.Series, pred: pd.Series) -> float
```

**Parameters:**
- `true` (pd.Series): Ground truth values
- `pred` (pd.Series): Predicted or imputed values

**Returns:**
- `float`: Mean absolute error

**Formula:**

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

**Example:**
```python
from imputation_showcase import mae

# Using same data as RMSE example
error_mae = mae(
    df_complete.loc[missing_mask, 'A'],
    df_imputed.loc[missing_mask, 'A']
)

print(f"MAE: {error_mae:.4f}")
```

**Use Cases:**
- More interpretable than RMSE
- When all errors should be weighted equally
- Presence of outliers in evaluation data

**Notes:**
- Units are same as original data
- More robust to outliers than RMSE
- Linear penalty for errors

---

## Usage Patterns

### Evaluating Single Method

```python
from imputation_showcase import KNNImputerMethod, rmse, mae
import numpy as np

# Introduce artificial missingness
mask = np.random.rand(*df_complete.shape) < 0.2
df_with_missing = df_complete.copy()
df_with_missing[mask] = np.nan

# Impute
imputer = KNNImputerMethod(k=5)
df_imputed = imputer.impute(df_with_missing)

# Extract values at missing locations
true_values = df_complete.values[mask]
imputed_values = df_imputed.values[mask]

# Evaluate
error_rmse = np.sqrt(np.mean((true_values - imputed_values) ** 2))
error_mae = np.mean(np.abs(true_values - imputed_values))

print(f"RMSE: {error_rmse:.4f}")
print(f"MAE: {error_mae:.4f}")
```

### Comparing Multiple Methods

```python
from imputation_showcase import (
    MeanImputer, KNNImputerMethod, MICEImputer,
    rmse, mae
)

methods = {
    'Mean': MeanImputer(),
    'KNN': KNNImputerMethod(k=5),
    'MICE': MICEImputer(random_state=42)
}

results = []
for name, imputer in methods.items():
    df_imputed = imputer.impute(df_with_missing)

    true_vals = df_complete.values[mask]
    imputed_vals = df_imputed.values[mask]

    results.append({
        'Method': name,
        'RMSE': np.sqrt(np.mean((true_vals - imputed_vals) ** 2)),
        'MAE': np.mean(np.abs(true_vals - imputed_vals))
    })

results_df = pd.DataFrame(results)
print(results_df.sort_values('RMSE'))
```

### Per-Column Evaluation

```python
def evaluate_per_column(df_true, df_imputed, mask):
    """Evaluate imputation quality for each column."""

    results = []
    for col in df_true.columns:
        col_mask = mask[col]
        if col_mask.sum() > 0:  # Has missing values
            col_rmse = rmse(
                df_true.loc[col_mask, col],
                df_imputed.loc[col_mask, col]
            )
            col_mae = mae(
                df_true.loc[col_mask, col],
                df_imputed.loc[col_mask, col]
            )
            results.append({
                'Column': col,
                'RMSE': col_rmse,
                'MAE': col_mae
            })

    return pd.DataFrame(results)

# Usage
evaluation = evaluate_per_column(df_complete, df_imputed, mask)
print(evaluation)
```

---

## Additional Metrics

While the library provides RMSE and MAE, you may want additional metrics:

### R² Score

```python
from sklearn.metrics import r2_score

r2 = r2_score(true_values, imputed_values)
print(f"R²: {r2:.4f}")
```

### Mean Absolute Percentage Error (MAPE)

```python
def mape(y_true, y_pred):
    """Calculate MAPE."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_score = mape(true_values, imputed_values)
print(f"MAPE: {mape_score:.2f}%")
```

### Distribution Comparison

```python
from scipy.stats import ks_2samp

# Kolmogorov-Smirnov test
observed_values = df_complete.values[~mask]
imputed_values = df_imputed.values[mask]

statistic, pvalue = ks_2samp(observed_values.flatten(), imputed_values.flatten())
print(f"KS statistic: {statistic:.4f}, p-value: {pvalue:.4f}")
```

---

## Best Practices

### 1. Evaluate Only on Missing Values

```python
# CORRECT: Evaluate only where values were actually missing
missing_mask = df_missing.isna()
error = rmse(
    df_complete[missing_mask],
    df_imputed[missing_mask]
)

# INCORRECT: Don't evaluate on all values
# error = rmse(df_complete, df_imputed)  # This includes observed values!
```

### 2. Use Holdout Set

```python
from sklearn.model_selection import train_test_split

# Split before introducing missingness
train_complete, test_complete = train_test_split(
    df_complete, test_size=0.2, random_state=42
)

# Introduce missingness in training set
train_missing = introduce_missingness(train_complete, rate=0.2)

# Fit imputer on training data
imputer = KNNImputerMethod(k=5)
train_imputed = imputer.impute(train_missing)

# Introduce missingness in test set (different pattern)
test_missing = introduce_missingness(test_complete, rate=0.2)
test_imputed = imputer.impute(test_missing)

# Evaluate on test set only
test_mask = test_missing.isna()
test_error = rmse(
    test_complete[test_mask],
    test_imputed[test_mask]
)
```

### 3. Report Multiple Metrics

```python
def comprehensive_evaluation(y_true, y_pred):
    """Return multiple evaluation metrics."""
    from sklearn.metrics import r2_score

    return {
        'RMSE': np.sqrt(np.mean((y_true - y_pred) ** 2)),
        'MAE': np.mean(np.abs(y_true - y_pred)),
        'R²': r2_score(y_true, y_pred),
        'Max Error': np.max(np.abs(y_true - y_pred)),
        'Median Error': np.median(np.abs(y_true - y_pred))
    }

metrics = comprehensive_evaluation(true_values, imputed_values)
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

---

## Type Hints

Functions use proper type hints:

```python
import pandas as pd

def rmse(true: pd.Series, pred: pd.Series) -> float:
    """Type-checked RMSE calculation."""
    ...

def mae(true: pd.Series, pred: pd.Series) -> float:
    """Type-checked MAE calculation."""
    ...
```

---

## Error Handling

```python
import pandas as pd
import numpy as np

try:
    error = rmse(true_series, pred_series)
except (ValueError, TypeError) as e:
    print(f"Error calculating RMSE: {e}")

# Handle NaN in evaluation
if pd.isna(error):
    print("RMSE is NaN - check for empty or all-NaN inputs")
```

---

## See Also

- [Imputation Methods API](methods.md)
- [Evaluation Guide](../user-guide/evaluation.md) for comprehensive evaluation strategies
- [Examples](../examples/ml-pipeline.md) for evaluation in ML pipelines
