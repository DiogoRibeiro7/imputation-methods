# Basic Concepts

This guide introduces the fundamental concepts of missing data imputation and how this library approaches the problem.

## What is Missing Data?

Missing data occurs when no value is stored for a variable in an observation. In pandas DataFrames, missing values are typically represented as `NaN` (Not a Number).

```python
import pandas as pd
import numpy as np

# Example of missing data
df = pd.DataFrame({
    'temperature': [20.5, 21.0, np.nan, 22.5, np.nan],
    'humidity': [65.0, np.nan, 70.0, 68.0, 72.0],
    'pressure': [1013, 1012, 1015, np.nan, 1014]
})

print(df)
#    temperature  humidity  pressure
# 0         20.5      65.0    1013.0
# 1         21.0       NaN    1012.0
# 2          NaN      70.0    1015.0
# 3         22.5      68.0       NaN
# 4          NaN      72.0    1014.0
```

## Types of Missing Data

Understanding the mechanism behind missing data is crucial for choosing the right imputation strategy:

### 1. Missing Completely at Random (MCAR)

The probability that a value is missing is independent of both observed and unobserved data.

**Example:** A sensor randomly fails due to hardware issues, unrelated to the measurements.

**Implications:** Safe to use most imputation methods. Simple methods like mean imputation work reasonably well.

### 2. Missing at Random (MAR)

The probability that a value is missing depends on observed data, but not on the missing value itself.

**Example:** Older patients are less likely to complete surveys, but conditional on age, missingness is random.

**Implications:** Can be handled with sophisticated methods that use relationships between variables (KNN, MICE, regression-based methods).

### 3. Missing Not at Random (MNAR)

The probability that a value is missing depends on the missing value itself.

**Example:** People with very high incomes often decline to report their salary.

**Implications:** Most difficult to handle. May require domain knowledge and specialized techniques beyond standard imputation.

## The Unified API

All imputation methods in this library inherit from `BaseImputer` and follow the same interface:

```python
from imputation_showcase import MeanImputer, KNNImputerMethod

# Every imputer has an impute() method
imputer = MeanImputer()
df_imputed = imputer.impute(df)

# Works the same way for all methods
knn_imputer = KNNImputerMethod(k=5)
df_imputed = knn_imputer.impute(df)
```

### Key Methods

#### `impute(df: pd.DataFrame) -> pd.DataFrame`

The primary method for all imputers. Takes a DataFrame with missing values and returns a complete DataFrame with imputed values.

**Parameters:**
- `df`: Pandas DataFrame with numeric columns containing `NaN` values

**Returns:**
- Pandas DataFrame with the same shape, with missing values replaced

**Requirements:**
- All columns must be numeric (float or int)
- Non-numeric columns should be encoded or removed before imputation

## When to Use Imputation

### Good Use Cases

1. **Statistical Analysis:** Missing data would reduce statistical power
2. **Machine Learning:** Most algorithms require complete data
3. **Time Series:** Continuous data needed for forecasting
4. **Reporting:** Complete data needed for visualization/dashboards

### When to Avoid Imputation

1. **High Missingness Rate:** If >40-50% of data is missing, imputation may introduce more bias than it resolves
2. **MNAR Data:** When missingness is informative, imputation can hide important patterns
3. **Small Datasets:** With few observations, imputation may not be reliable
4. **Causal Inference:** Imputation can complicate causal relationships

## Trade-offs in Imputation

### Bias vs. Variance

- **Simple methods** (mean, median): Low variance, potentially high bias
- **Complex methods** (MICE, deep learning): Lower bias, potentially higher variance

### Computational Cost

- **Fast methods** (mean, median): O(n) complexity
- **Moderate methods** (KNN, regression): O(n²) or O(n log n) complexity
- **Slow methods** (MICE, MissForest, neural networks): Iterative, can be very slow

### Interpretability

- **Interpretable:** Mean, median, LOCF - easy to understand and explain
- **Moderate:** Regression, KNN - somewhat interpretable
- **Black box:** Autoencoders, GAIN - difficult to interpret

## Evaluation Strategy

When ground truth is available, you can evaluate imputation quality:

```python
from imputation_showcase import KNNImputerMethod, rmse, mae
import numpy as np

# Original complete data
df_complete = pd.DataFrame({
    'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
    'feature2': [5.0, 6.0, 7.0, 8.0, 10.0]
})

# Introduce artificial missingness
df_missing = df_complete.copy()
mask = np.array([[False, True], [True, False], [False, False], [True, True], [False, True]])
df_missing[mask] = np.nan

# Impute
imputer = KNNImputerMethod(k=2)
df_imputed = imputer.impute(df_missing)

# Evaluate only on originally missing values
original_values = df_complete.values[mask]
imputed_values = df_imputed.values[mask]

error_rmse = np.sqrt(np.mean((original_values - imputed_values) ** 2))
error_mae = np.mean(np.abs(original_values - imputed_values))

print(f"RMSE: {error_rmse:.4f}")
print(f"MAE: {error_mae:.4f}")
```

## Best Practices

### 1. Understand Your Data

Before imputation, analyze the missing data pattern:

```python
import missingno as msno
import matplotlib.pyplot as plt

# Visualize missingness pattern
msno.matrix(df)
plt.show()

# Check missingness percentage
missing_pct = df.isna().sum() / len(df) * 100
print(missing_pct)
```

### 2. Split Data First

Always split into train/test sets **before** imputation to avoid data leakage:

```python
from sklearn.model_selection import train_test_split

# CORRECT: Split first
X_train, X_test = train_test_split(X, test_size=0.2)

# Fit imputer on training data only
imputer = KNNImputerMethod(k=5)
X_train_imputed = imputer.impute(X_train)
X_test_imputed = imputer.impute(X_test)

# INCORRECT: Don't impute before splitting
# X_imputed = imputer.impute(X)  # This causes data leakage!
# X_train, X_test = train_test_split(X_imputed, test_size=0.2)
```

### 3. Try Multiple Methods

Different methods work better for different data types and missingness patterns:

```python
from imputation_showcase import MeanImputer, KNNImputerMethod, MICEImputer

methods = {
    'Mean': MeanImputer(),
    'KNN': KNNImputerMethod(k=5),
    'MICE': MICEImputer()
}

for name, imputer in methods.items():
    df_imputed = imputer.impute(df)
    # Evaluate each method
    print(f"{name}: {evaluate(df_imputed)}")
```

### 4. Document Your Choices

Always document which imputation method you used and why:

```python
# Good practice: Document your imputation strategy
IMPUTATION_CONFIG = {
    'method': 'KNN',
    'parameters': {'k': 5},
    'rationale': 'KNN chosen because data has clear feature correlations',
    'date': '2024-11-20'
}
```

## Next Steps

Now that you understand the basic concepts:

- Explore the [User Guide](../user-guide/methods.md) for detailed method descriptions
- Learn about [Method Selection](../user-guide/selection.md) to choose the right approach
- See [Examples](../examples/time-series.md) for real-world use cases
- Check the [API Reference](../api/methods.md) for technical details
