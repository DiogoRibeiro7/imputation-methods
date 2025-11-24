# Evaluation Metrics

Learn how to evaluate imputation quality and compare different methods using quantitative metrics and visualizations.

## Why Evaluate Imputation?

Imputation is rarely an end goal—it's a preprocessing step. However, evaluating imputation quality helps you:

1. **Choose the best method** for your data
2. **Detect issues** with imputation strategy
3. **Validate assumptions** about missing data
4. **Document methodology** for reproducibility
5. **Monitor performance** in production

## Evaluation Strategies

### 1. Direct Evaluation (Ground Truth Available)

When you have complete data, artificially introduce missingness to evaluate imputation quality.

```python
import numpy as np
import pandas as pd
from imputation_showcase import KNNImputerMethod, rmse, mae

# Original complete dataset
df_complete = pd.DataFrame({
    'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'feature2': [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
})

# Introduce 20% missingness randomly
np.random.seed(42)
mask = np.random.rand(*df_complete.shape) < 0.2
df_with_missing = df_complete.copy()
df_with_missing[mask] = np.nan

# Impute
imputer = KNNImputerMethod(k=3)
df_imputed = imputer.impute(df_with_missing)

# Evaluate only on artificially missing values
original_values = df_complete.values[mask]
imputed_values = df_imputed.values[mask]

error_rmse = np.sqrt(np.mean((original_values - imputed_values) ** 2))
error_mae = np.mean(np.abs(original_values - imputed_values))

print(f"RMSE: {error_rmse:.4f}")
print(f"MAE: {error_mae:.4f}")
```

### 2. Indirect Evaluation (No Ground Truth)

When ground truth isn't available, evaluate based on downstream task performance.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from imputation_showcase import MeanImputer, KNNImputerMethod, MICEImputer

# Your data with missing values (no ground truth)
X_missing = load_your_data()
y = load_your_labels()

methods = {
    'Mean': MeanImputer(),
    'KNN': KNNImputerMethod(k=5),
    'MICE': MICEImputer(random_state=42)
}

results = {}
for name, imputer in methods.items():
    X_imputed = imputer.impute(X_missing)
    model = RandomForestRegressor(random_state=42)
    scores = cross_val_score(model, X_imputed, y, cv=5, scoring='r2')
    results[name] = scores.mean()
    print(f"{name}: R² = {scores.mean():.4f} (±{scores.std():.4f})")

# Choose method with best downstream performance
best_method = max(results, key=results.get)
print(f"\nBest method: {best_method}")
```

### 3. Distribution-Based Evaluation

Compare distributions of imputed values to observed values.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def compare_distributions(df_original, df_missing, df_imputed, column):
    """Compare distributions visually."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Original distribution
    axes[0].hist(df_original[column].dropna(), bins=30, alpha=0.7, color='blue')
    axes[0].set_title('Original Distribution')
    axes[0].set_ylabel('Frequency')

    # Observed vs Missing locations
    missing_mask = df_missing[column].isna()
    axes[1].hist(df_original.loc[~missing_mask, column], bins=30,
                 alpha=0.7, label='Observed', color='blue')
    axes[1].hist(df_original.loc[missing_mask, column], bins=30,
                 alpha=0.7, label='Was Missing', color='red')
    axes[1].set_title('Observed vs. Missing Locations')
    axes[1].legend()

    # Observed vs Imputed
    axes[2].hist(df_missing[column].dropna(), bins=30,
                 alpha=0.7, label='Observed', color='blue')
    axes[2].hist(df_imputed.loc[missing_mask, column], bins=30,
                 alpha=0.7, label='Imputed', color='green')
    axes[2].set_title('Observed vs. Imputed Values')
    axes[2].legend()

    plt.tight_layout()
    return fig

# Usage
fig = compare_distributions(df_complete, df_with_missing, df_imputed, 'feature1')
plt.show()
```

## Metrics

### Root Mean Squared Error (RMSE)

Measures average magnitude of errors, with higher weight on large errors.

```python
from imputation_showcase import rmse

# For pandas Series
error = rmse(true_series, imputed_series)

# For numpy arrays
error = np.sqrt(np.mean((y_true - y_pred) ** 2))
```

**Interpretation:**
- Lower is better
- Same units as original data
- Sensitive to outliers
- Penalizes large errors more than MAE

**When to use:**
- When large errors are particularly undesirable
- Comparing methods on same dataset
- Data is approximately normal

### Mean Absolute Error (MAE)

Measures average absolute difference between true and imputed values.

```python
from imputation_showcase import mae

# For pandas Series
error = mae(true_series, imputed_series)

# For numpy arrays
error = np.mean(np.abs(y_true - y_pred))
```

**Interpretation:**
- Lower is better
- Same units as original data
- More robust to outliers than RMSE
- Linear penalty for errors

**When to use:**
- When all errors are equally important
- Presence of outliers
- Want more interpretable metric

### Mean Absolute Percentage Error (MAPE)

Percentage-based error metric.

```python
def mape(y_true, y_pred):
    """Calculate MAPE."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

error = mape(true_values, imputed_values)
print(f"MAPE: {error:.2f}%")
```

**Interpretation:**
- Scale-independent (useful for comparing across datasets)
- Percentage makes it intuitive
- **Warning:** Undefined when true value is 0
- Biased toward under-predictions

**When to use:**
- Comparing across different scales
- Business-oriented reporting
- Values are always positive and non-zero

### R² Score (Coefficient of Determination)

Measures proportion of variance explained by imputation.

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.4f}")
```

**Interpretation:**
- Range: (-∞, 1], where 1 is perfect
- 0 means imputation is as good as mean
- Negative means imputation is worse than mean
- Scale-free

**When to use:**
- Want to know proportion of variance captured
- Comparing different datasets
- Statistical reporting

## Comprehensive Evaluation Framework

Complete evaluation comparing multiple methods:

```python
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from imputation_showcase import (
    MeanImputer, MedianImputer, KNNImputerMethod,
    MICEImputer, MissForestImputer, rmse, mae
)

def evaluate_imputation_methods(df_complete, missing_rate=0.2, random_state=42):
    """
    Comprehensive evaluation of imputation methods.

    Args:
        df_complete: Complete dataset (ground truth)
        missing_rate: Proportion of values to make missing
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with evaluation metrics for each method
    """
    # Create missing data
    np.random.seed(random_state)
    mask = np.random.rand(*df_complete.shape) < missing_rate
    df_missing = df_complete.copy()
    df_missing[mask] = np.nan

    # Define methods to evaluate
    methods = {
        'Mean': MeanImputer(),
        'Median': MedianImputer(),
        'KNN-3': KNNImputerMethod(k=3),
        'KNN-5': KNNImputerMethod(k=5),
        'KNN-7': KNNImputerMethod(k=7),
        'MICE': MICEImputer(random_state=random_state),
        'MissForest': MissForestImputer(random_state=random_state)
    }

    results = []

    for method_name, imputer in methods.items():
        print(f"Evaluating {method_name}...")

        # Time the imputation
        import time
        start_time = time.time()
        df_imputed = imputer.impute(df_missing)
        elapsed_time = time.time() - start_time

        # Extract values
        true_values = df_complete.values[mask]
        imputed_values = df_imputed.values[mask]

        # Calculate metrics
        error_rmse = np.sqrt(np.mean((true_values - imputed_values) ** 2))
        error_mae = np.mean(np.abs(true_values - imputed_values))
        r2 = r2_score(true_values, imputed_values)

        # Calculate distribution similarity (KS test)
        from scipy.stats import ks_2samp
        observed_values = df_missing.values[~mask]
        _, ks_pvalue = ks_2samp(observed_values, imputed_values)

        results.append({
            'Method': method_name,
            'RMSE': error_rmse,
            'MAE': error_mae,
            'R²': r2,
            'KS p-value': ks_pvalue,
            'Time (s)': elapsed_time
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE')

    return results_df, df_missing, {name: imputer.impute(df_missing)
                                     for name, imputer in methods.items()}

# Usage
results_df, df_missing, imputed_dfs = evaluate_imputation_methods(df_complete)
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)
print(results_df.to_string(index=False))
```

## Visualization

### 1. Error Distribution

Visualize how errors are distributed:

```python
def plot_error_distribution(y_true, y_pred, method_name):
    """Plot error distribution."""
    errors = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of errors
    axes[0].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Error (Imputed - True)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{method_name}: Error Distribution')

    # QQ plot
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title(f'{method_name}: Q-Q Plot')

    plt.tight_layout()
    return fig
```

### 2. Scatter Plot: True vs. Imputed

```python
def plot_true_vs_imputed(y_true, y_pred, method_name):
    """Scatter plot of true vs imputed values."""

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            'r--', lw=2, label='Perfect prediction')

    ax.set_xlabel('True Values')
    ax.set_ylabel('Imputed Values')
    ax.set_title(f'{method_name}: True vs. Imputed')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add R² annotation
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}',
            transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return fig
```

### 3. Method Comparison Heatmap

```python
def plot_comparison_heatmap(results_df):
    """Heatmap comparing methods across metrics."""

    # Normalize metrics to [0, 1] where 1 is best
    metrics = ['RMSE', 'MAE', 'R²', 'Time (s)']

    normalized = results_df[metrics].copy()
    # For RMSE, MAE, Time: lower is better
    for col in ['RMSE', 'MAE', 'Time (s)']:
        normalized[col] = 1 - (normalized[col] - normalized[col].min()) / \
                          (normalized[col].max() - normalized[col].min())
    # For R²: higher is better (already normalized)
    normalized['R²'] = (normalized['R²'] - normalized['R²'].min()) / \
                       (normalized['R²'].max() - normalized['R²'].min())

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(normalized.T, annot=True, fmt='.3f',
                cmap='RdYlGn', center=0.5,
                xticklabels=results_df['Method'],
                yticklabels=metrics,
                cbar_kws={'label': 'Normalized Score\n(1=Best)'})
    ax.set_title('Method Comparison Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
```

## Cross-Validation for Imputation

Properly evaluate imputation in a cross-validation setting:

```python
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

def cv_evaluate_imputation(X, y, imputer, model, cv=5):
    """
    Cross-validation evaluation of imputation + modeling.

    Args:
        X: Features with missing values
        y: Target variable
        imputer: Imputation method
        model: ML model
        cv: Number of folds

    Returns:
        Array of CV scores
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Impute (fit on train, transform both)
        X_train_imputed = imputer.impute(X_train)
        X_test_imputed = imputer.impute(X_test)

        # Train and evaluate model
        model.fit(X_train_imputed, y_train)
        score = model.score(X_test_imputed, y_test)
        scores.append(score)

        print(f"Fold {fold+1}: R² = {score:.4f}")

    scores = np.array(scores)
    print(f"\nMean: {scores.mean():.4f} (±{scores.std():.4f})")
    return scores

# Usage
from imputation_showcase import KNNImputerMethod

imputer = KNNImputerMethod(k=5)
model = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cv_evaluate_imputation(X, y, imputer, model, cv=5)
```

## Production Monitoring

Monitor imputation quality in production:

```python
class ImputationMonitor:
    """Monitor imputation patterns in production."""

    def __init__(self, alert_threshold=0.3):
        self.alert_threshold = alert_threshold
        self.history = []

    def log_batch(self, df_before, df_after):
        """Log statistics for a batch."""
        stats = {
            'timestamp': pd.Timestamp.now(),
            'n_missing': df_before.isna().sum().sum(),
            'missing_rate': df_before.isna().sum().sum() / df_before.size,
            'mean_before': df_before.mean().mean(),
            'mean_after': df_after.mean().mean(),
            'std_before': df_before.std().mean(),
            'std_after': df_after.std().mean()
        }

        self.history.append(stats)

        # Alert if missing rate is high
        if stats['missing_rate'] > self.alert_threshold:
            self.alert(f"High missing rate: {stats['missing_rate']:.2%}")

        return stats

    def alert(self, message):
        """Send alert (implement your alerting logic)."""
        print(f"⚠️  ALERT: {message}")

    def plot_trends(self):
        """Plot imputation trends over time."""
        df = pd.DataFrame(self.history)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(df['timestamp'], df['missing_rate'], marker='o')
        axes[0].axhline(self.alert_threshold, color='red',
                        linestyle='--', label='Alert Threshold')
        axes[0].set_ylabel('Missing Rate')
        axes[0].set_title('Missing Data Rate Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df['timestamp'], df['mean_before'], marker='o', label='Before')
        axes[1].plot(df['timestamp'], df['mean_after'], marker='s', label='After')
        axes[1].set_ylabel('Mean Value')
        axes[1].set_xlabel('Time')
        axes[1].set_title('Mean Values Before/After Imputation')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

# Usage
monitor = ImputationMonitor(alert_threshold=0.3)

# In production loop
for batch in data_stream:
    imputed_batch = imputer.impute(batch)
    monitor.log_batch(batch, imputed_batch)

# Periodically review trends
monitor.plot_trends()
```

## Best Practices

1. **Always use holdout test set**: Evaluate on data not seen during imputer fitting
2. **Stratified evaluation**: Evaluate across different subgroups
3. **Multiple metrics**: Don't rely on a single metric
4. **Visual inspection**: Always look at distributions and scatter plots
5. **Domain validation**: Check if imputed values make sense
6. **Monitor in production**: Track missingness patterns over time

## Next Steps

- Review [Best Practices](best-practices.md) for production deployment
- See [Examples](../examples/time-series.md) for complete workflows
- Check [Method Selection](selection.md) to choose the right approach
- Consult [API Reference](../api/evaluation.md) for function details
