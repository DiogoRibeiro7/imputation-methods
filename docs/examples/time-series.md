# Time Series Imputation Example

This example demonstrates imputation techniques specifically designed for time series data, such as sensor readings and temporal datasets.

## Overview

Time series data has unique characteristics that require specialized imputation approaches:

- **Temporal ordering** matters - values evolve over time
- **Temporal dependency** - current values depend on past values
- **Seasonal patterns** - recurring patterns over time periods
- **Trend** - long-term directional movement

## Full Example

For a complete, runnable implementation, see `examples/time_series_example.py` in the repository.

## Problem Description

**Scenario:** Sensor network collecting temperature, humidity, and pressure readings every hour. Sensors occasionally fail, creating gaps in the data.

**Challenge:** Fill missing sensor readings while preserving temporal patterns and relationships between sensors.

## Quick Start

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from imputation_showcase import LOCFImputer, NOCBImputer, KNNImputerMethod

# Generate synthetic sensor data
timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(168)]  # 1 week
temperature = 20 + 5 * np.sin(2 * np.pi * np.arange(168) / 24) + np.random.normal(0, 0.5, 168)
humidity = 60 - 10 * np.sin(2 * np.pi * np.arange(168) / 24) + np.random.normal(0, 2, 168)

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'temperature': temperature,
    'humidity': humidity
})

# Introduce 15% missingness
np.random.seed(42)
for col in ['temperature', 'humidity']:
    mask = np.random.rand(len(df)) < 0.15
    df.loc[mask, col] = np.nan

print(f"Missing values: {df.isna().sum().sum()}")
```

## Method Comparison

### 1. LOCF (Last Observation Carried Forward)

Best for slowly changing variables.

```python
from imputation_showcase import LOCFImputer

# Separate numeric columns for imputation
numeric_cols = ['temperature', 'humidity']
df_numeric = df[numeric_cols].copy()

# Apply LOCF
locf_imputer = LOCFImputer()
df_locf = locf_imputer.impute(df_numeric)

# Add timestamp back
df_locf['timestamp'] = df['timestamp']

print("LOCF Imputation Complete")
print(df_locf.head(10))
```

**When to use LOCF:**
- Slowly changing variables (temperature, pressure)
- High-frequency sampling (seconds/minutes)
- Forward causality is acceptable
- Real-time applications

**Limitations:**
- Can introduce bias in long gaps
- Not suitable for rapidly changing variables
- First missing value cannot be filled

### 2. NOCB (Next Observation Carried Backward)

Useful for backward-looking analysis.

```python
from imputation_showcase import NOCBImputer

nocb_imputer = NOCBImputer()
df_nocb = nocb_imputer.impute(df_numeric)
df_nocb['timestamp'] = df['timestamp']

print("NOCB Imputation Complete")
```

**When to use NOCB:**
- Retrospective analysis
- Filling gaps before a known event
- Complementary to LOCF

**Limitations:**
- Last missing value cannot be filled
- Can introduce look-ahead bias
- Less intuitive than forward filling

### 3. KNN for Time Series

Considers relationships between multiple sensors.

```python
from imputation_showcase import KNNImputerMethod

# KNN works well when multiple correlated sensors exist
knn_imputer = KNNImputerMethod(k=5)
df_knn = knn_imputer.impute(df_numeric)
df_knn['timestamp'] = df['timestamp']

print("KNN Imputation Complete")
```

**When to use KNN for time series:**
- Multiple correlated sensors
- Medium-frequency sampling (minutes/hours)
- Higher accuracy needed
- Batch processing acceptable

**Advantages:**
- Leverages cross-sensor correlations
- More accurate than simple methods
- Preserves relationships

**Limitations:**
- Computationally expensive
- Requires sufficient historical data
- Sensitive to feature scaling

## Evaluation

Compare methods using metrics:

```python
from imputation_showcase import rmse, mae

# Assuming we have ground truth for validation
methods = {
    'LOCF': df_locf,
    'NOCB': df_nocb,
    'KNN': df_knn
}

print("\nEvaluation Results:")
print(f"{'Method':<10s} {'Temp RMSE':>10s} {'Humid RMSE':>11s}")
print("-" * 35)

for method_name, imputed_df in methods.items():
    temp_rmse = rmse(df_complete['temperature'], imputed_df['temperature'])
    humid_rmse = rmse(df_complete['humidity'], imputed_df['humidity'])
    print(f"{method_name:<10s} {temp_rmse:10.4f} {humid_rmse:11.4f}")
```

## Visualization

Visualize the results:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot each method
for idx, (method_name, imputed_df) in enumerate(methods.items()):
    ax = axes[idx]

    # Plot original data
    ax.plot(df['timestamp'], df['temperature'], 'b--',
            label='Observed', alpha=0.5, linewidth=1)

    # Plot imputed data
    ax.plot(imputed_df['timestamp'], imputed_df['temperature'],
            'g-', label='Imputed', alpha=0.8, linewidth=2)

    # Highlight missing locations
    missing_mask = df['temperature'].isna()
    ax.scatter(df.loc[missing_mask, 'timestamp'],
              imputed_df.loc[missing_mask, 'temperature'],
              color='red', s=50, label='Imputed values', zorder=5)

    ax.set_title(f'{method_name} Imputation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperature (°C)')
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.savefig('time_series_comparison.png', dpi=150)
plt.show()
```

## Advanced: Rolling Window Imputation

For more sophisticated time series imputation:

```python
def rolling_mean_impute(df, window=5):
    """Impute using rolling mean."""
    df_imputed = df.copy()

    for col in df.columns:
        # Calculate rolling mean on observed values
        rolling_mean = df[col].rolling(window=window, min_periods=1).mean()

        # Fill missing values with rolling mean
        df_imputed[col] = df[col].fillna(rolling_mean)

    return df_imputed

# Apply rolling window imputation
df_rolling = rolling_mean_impute(df_numeric, window=5)
df_rolling['timestamp'] = df['timestamp']
```

## Real-World Considerations

### 1. Handling Long Gaps

```python
def impute_with_gap_limit(df, imputer, max_gap=3):
    """Impute only if gap is smaller than max_gap."""
    df_imputed = df.copy()

    for col in df.columns:
        # Identify gaps
        is_missing = df[col].isna()
        gap_sizes = is_missing.groupby((~is_missing).cumsum()).cumsum()

        # Only impute small gaps
        small_gaps = gap_sizes <= max_gap
        df_to_impute = df[[col]].copy()
        df_to_impute.loc[is_missing & ~small_gaps, col] = np.nan

        # Impute small gaps
        df_imputed[col] = imputer.impute(df_to_impute)[col]

    return df_imputed

# Usage
df_smart = impute_with_gap_limit(df_numeric, LOCFImputer(), max_gap=3)
```

### 2. Multivariate Time Series

```python
# For multiple correlated time series
from imputation_showcase import MICEImputer

# MICE can handle complex dependencies
mice_imputer = MICEImputer(random_state=42)
df_mice = mice_imputer.impute(df_numeric)
```

### 3. Seasonal Patterns

```python
def seasonal_impute(df, period=24):
    """Impute using seasonal patterns."""
    df_imputed = df.copy()

    for col in df.columns:
        for i in range(len(df)):
            if pd.isna(df.loc[i, col]):
                # Find values at same time in previous periods
                same_time_values = df.loc[
                    df.index % period == i % period, col
                ].dropna()

                if len(same_time_values) > 0:
                    df_imputed.loc[i, col] = same_time_values.median()

    return df_imputed

# Usage
df_seasonal = seasonal_impute(df_numeric, period=24)  # Daily pattern
```

## Key Takeaways

1. **LOCF** is simple and fast, best for slowly changing variables
2. **KNN** provides better accuracy when multiple sensors are available
3. **Consider temporal patterns** - daily, weekly, seasonal
4. **Limit gap size** - don't impute very long gaps
5. **Validate thoroughly** - check if imputed patterns make sense
6. **Domain knowledge** is crucial for time series

## Complete Example Code

Run the full example with visualizations:

```bash
cd examples/
python time_series_example.py
```

This will generate:
- Comparison plots for different methods
- Performance metrics
- Detailed analysis of a specific time window

## Next Steps

- Explore [ML Pipeline Integration](ml-pipeline.md) for using imputed data in models
- Learn about [Custom Imputers](custom-imputers.md) to implement domain-specific logic
- Review [Best Practices](../user-guide/best-practices.md) for production deployment
