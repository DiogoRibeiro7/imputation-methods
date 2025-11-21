# Quick Start

This guide will help you perform your first imputation in just a few minutes.

## Your First Imputation

### Step 1: Import Libraries

```python
import pandas as pd
import numpy as np
from imputation_showcase.imputation_methods import MeanImputer
```

### Step 2: Create Data with Missing Values

```python
# Create a simple dataset
df = pd.DataFrame({
    'temperature': [20.5, 21.3, np.nan, 19.8, 22.1],
    'humidity': [65, 70, np.nan, 72, 69],
    'pressure': [1013, np.nan, 1015, 1014, 1016]
})

print("Original data:")
print(df)
print(f"\nMissing values: {df.isna().sum().sum()}")
```

Output:
```
   temperature  humidity  pressure
0         20.5      65.0    1013.0
1         21.3      70.0       NaN
2          NaN       NaN    1015.0
3         19.8      72.0    1014.0
4         22.1      69.0    1016.0

Missing values: 3
```

### Step 3: Apply Imputation

```python
# Create imputer
imputer = MeanImputer()

# Impute missing values
df_imputed = imputer.impute(df)

print("Imputed data:")
print(df_imputed)
print(f"\nMissing values: {df_imputed.isna().sum().sum()}")
```

Output:
```
   temperature   humidity  pressure
0    20.500000  65.000000   1013.00
1    21.300000  70.000000   1014.50
2    20.925000  69.000000   1015.00
3    19.800000  72.000000   1014.00
4    22.100000  69.000000   1016.00

Missing values: 0
```

## Trying Different Methods

### K-Nearest Neighbors

For more accurate imputation using similar observations:

```python
from imputation_showcase.imputation_methods import KNNImputerMethod

# Use 3 nearest neighbors
knn_imputer = KNNImputerMethod(k=3)
df_knn = knn_imputer.impute(df)

print(df_knn)
```

### Median Imputation

For data with outliers:

```python
from imputation_showcase.imputation_methods import MedianImputer

median_imputer = MedianImputer()
df_median = median_imputer.impute(df)

print(df_median)
```

## Evaluating Imputation Quality

When you have ground truth (original complete data):

```python
from imputation_showcase.imputation_methods import rmse, mae

# Assume df_complete is your original data before introducing missing values
df_complete = pd.DataFrame({
    'temperature': [20.5, 21.3, 20.9, 19.8, 22.1],
    'humidity': [65, 70, 68, 72, 69],
    'pressure': [1013, 1014.5, 1015, 1014, 1016]
})

# Calculate error metrics
error_rmse = rmse(df_complete, df_imputed)
error_mae = mae(df_complete, df_imputed)

print(f"RMSE: {error_rmse:.4f}")
print(f"MAE: {error_mae:.4f}")
```

## Comparing Multiple Methods

```python
from imputation_showcase.imputation_methods import (
    MeanImputer,
    MedianImputer,
    KNNImputerMethod,
    rmse
)

# Define methods to compare
methods = {
    'Mean': MeanImputer(),
    'Median': MedianImputer(),
    'KNN-3': KNNImputerMethod(k=3),
    'KNN-5': KNNImputerMethod(k=5),
}

# Compare all methods
for name, imputer in methods.items():
    df_imp = imputer.impute(df)
    error = rmse(df_complete, df_imp)
    print(f"{name:10s} RMSE: {error:.4f}")
```

## Working with Real Data

### Loading from CSV

```python
# Load your data
df = pd.read_csv("data/your_data.csv")

# Check for missing values
print(f"Missing values per column:")
print(df.isna().sum())

# Apply imputation
imputer = KNNImputerMethod(k=5)
df_imputed = imputer.impute(df)

# Save results
df_imputed.to_csv("data/imputed_data.csv", index=False)
```

### Integration with Machine Learning

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from imputation_showcase.imputation_methods import KNNImputerMethod

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Impute training data
imputer = KNNImputerMethod(k=5)
X_train_imputed = imputer.impute(X_train)
X_test_imputed = imputer.impute(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_imputed, y_train)

# Predict
predictions = model.predict(X_test_imputed)
```

## Best Practices

!!! tip "Data Validation"
    Always ensure your data contains only numeric columns before imputation:
    ```python
    # Check data types
    print(df.dtypes)

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    ```

!!! warning "Train/Test Split"
    When using imputation in ML pipelines, fit the imputer on training data only:
    ```python
    # ✓ Correct
    imputer.fit(X_train)  # Fit on training data
    X_train_imp = imputer.transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # ✗ Incorrect (data leakage)
    imputer.fit(X)  # Don't fit on combined data
    ```

!!! info "Missing Data Pattern"
    Understand your missing data pattern before choosing a method:
    - **MCAR** (Missing Completely at Random): Any method works
    - **MAR** (Missing at Random): Use KNN, MICE
    - **MNAR** (Missing Not at Random): Advanced methods recommended

## Next Steps

- [Basic Concepts](concepts.md) - Learn about missing data patterns
- [Methods Overview](../user-guide/methods.md) - Explore all 16+ methods
- [Examples](../examples/time-series.md) - See practical use cases
- [Best Practices](../user-guide/best-practices.md) - Production guidelines
