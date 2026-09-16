# ML Pipeline Integration

Learn how to integrate imputation into complete machine learning pipelines for production use.

## Overview

Real-world ML systems need robust imputation as part of the preprocessing pipeline. This example shows how to:

- Integrate imputation with scikit-learn pipelines
- Avoid data leakage
- Compare imputation methods for downstream tasks
- Deploy imputation in production

## Full Example

For a complete, runnable implementation, see [`examples/ml_pipeline_example.py`](https://github.com/DiogoRibeiro7/imputation-methods/blob/main/examples/ml_pipeline_example.py) in the repository.

## Problem Description

**Scenario:** Predicting house prices with a dataset containing missing values in features like square footage, number of bedrooms, age, etc.

**Goal:** Build an end-to-end ML pipeline that handles missing data and produces accurate predictions.

## Quick Start

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from imputation_methods import KNNImputerMethod, MICEImputer, MeanImputer

# Load data with missing values
df = pd.read_csv('housing_data.csv')
X = df.drop('price', axis=1)
y = df['price']

# Split BEFORE imputation (critical!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Impute
imputer = KNNImputerMethod(k=5)
X_train_imputed = imputer.impute(X_train)
X_test_imputed = imputer.impute(X_test)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Train model
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Test R²: {r2:.4f}")
print(f"Test RMSE: ${rmse:,.2f}")
```

## Building a Pipeline Class

Create a reusable pipeline:

```python
class ImputedPipeline:
    """ML Pipeline with integrated imputation."""

    def __init__(self, imputer, model, scaler=True):
        """
        Initialize pipeline.

        Args:
            imputer: Imputation method from imputation_methods
            model: ML model (sklearn-compatible)
            scaler: Whether to apply StandardScaler
        """
        self.imputer = imputer
        self.model = model
        self.scaler = StandardScaler() if scaler else None

    def fit(self, X, y):
        """Fit the pipeline."""
        # Step 1: Impute missing values
        X_imputed = self.imputer.impute(X)

        # Step 2: Scale features
        if self.scaler:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_imputed),
                columns=X_imputed.columns,
                index=X_imputed.index
            )
        else:
            X_scaled = X_imputed

        # Step 3: Train model
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        """Make predictions."""
        # Apply same transformations
        X_imputed = self.imputer.impute(X)

        if self.scaler:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_imputed),
                columns=X_imputed.columns,
                index=X_imputed.index
            )
        else:
            X_scaled = X_imputed

        return self.model.predict(X_scaled)

    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

# Usage
pipeline = ImputedPipeline(
    imputer=KNNImputerMethod(k=5),
    model=Ridge(alpha=1.0),
    scaler=True
)

pipeline.fit(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
print(f"Test R²: {test_score:.4f}")
```

## Comparing Imputation Methods

Evaluate multiple imputation strategies:

```python
from imputation_methods import (
    MeanImputer, MedianImputer, KNNImputerMethod,
    MICEImputer, MissForestImputer
)
from sklearn.ensemble import RandomForestRegressor

# Define methods to compare
imputation_methods = {
    'Mean + Ridge': (MeanImputer(), Ridge(alpha=1.0)),
    'Median + Ridge': (MedianImputer(), Ridge(alpha=1.0)),
    'KNN + Ridge': (KNNImputerMethod(k=5), Ridge(alpha=1.0)),
    'MICE + Ridge': (MICEImputer(random_state=42), Ridge(alpha=1.0)),
    'KNN + RF': (KNNImputerMethod(k=5), RandomForestRegressor(n_estimators=100, random_state=42)),
}

# Evaluate each combination
results = []

for name, (imputer, model) in imputation_methods.items():
    print(f"Evaluating {name}...")

    pipeline = ImputedPipeline(imputer, model, scaler=True)
    pipeline.fit(X_train, y_train)

    # Metrics
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    results.append({
        'Method': name,
        'Train R²': train_score,
        'Test R²': test_score,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse
    })

# Display results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test R²', ascending=False)

print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)
print(results_df.to_string(index=False))

# Find best method
best_method = results_df.iloc[0]['Method']
best_score = results_df.iloc[0]['Test R²']
print(f"\nBest Method: {best_method}")
print(f"Test R²: {best_score:.4f}")
```

## Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparison(results_df):
    """Plot comparison of imputation methods."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Test RMSE comparison
    axes[0, 0].barh(results_df['Method'], results_df['Test RMSE'])
    axes[0, 0].set_xlabel('Test RMSE ($)')
    axes[0, 0].set_title('Test RMSE (Lower is Better)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # Test R² comparison
    axes[0, 1].barh(results_df['Method'], results_df['Test R²'], color='green')
    axes[0, 1].set_xlabel('Test R²')
    axes[0, 1].set_title('Test R² (Higher is Better)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # Train vs Test RMSE
    x = np.arange(len(results_df))
    width = 0.35
    axes[1, 0].bar(x - width/2, results_df['Train RMSE'], width, label='Train', alpha=0.8)
    axes[1, 0].bar(x + width/2, results_df['Test RMSE'], width, label='Test', alpha=0.8)
    axes[1, 0].set_ylabel('RMSE ($)')
    axes[1, 0].set_title('Train vs Test RMSE', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(results_df['Method'], rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Overfitting analysis
    axes[1, 1].scatter(results_df['Train R²'], results_df['Test R²'], s=100, alpha=0.6)
    for idx, row in results_df.iterrows():
        axes[1, 1].annotate(row['Method'], (row['Train R²'], row['Test R²']),
                           fontsize=8, ha='left', va='bottom')
    axes[1, 1].plot([0.7, 1], [0.7, 1], 'r--', alpha=0.5, label='No overfitting')
    axes[1, 1].set_xlabel('Train R²')
    axes[1, 1].set_ylabel('Test R²')
    axes[1, 1].set_title('Overfitting Analysis', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

fig = plot_comparison(results_df)
plt.savefig('ml_pipeline_comparison.png', dpi=150)
plt.show()
```

## Scikit-learn Integration

Use with scikit-learn's Pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class ImputationTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible imputation transformer."""

    def __init__(self, imputer):
        self.imputer = imputer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.imputer.impute(X)

# Create sklearn pipeline
from sklearn.ensemble import GradientBoostingRegressor

pipeline = Pipeline([
    ('impute', ImputationTransformer(KNNImputerMethod(k=5))),
    ('scale', StandardScaler()),
    ('model', GradientBoostingRegressor(random_state=42))
])

# Use like any sklearn model
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

# Can be used with GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
```

## Cross-Validation

Properly evaluate with cross-validation:

```python
from sklearn.model_selection import cross_val_score

def cv_evaluate_imputation(X, y, imputer, model, cv=5):
    """Cross-validation with proper imputation."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Split data
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]

        # Create and fit pipeline
        pipeline = ImputedPipeline(imputer, model, scaler=True)
        pipeline.fit(X_train_fold, y_train_fold)

        # Evaluate
        score = pipeline.score(X_test_fold, y_test_fold)
        scores.append(score)

    return np.array(scores)

# Evaluate with CV
scores = cv_evaluate_imputation(
    X, y,
    imputer=KNNImputerMethod(k=5),
    model=Ridge(alpha=1.0),
    cv=5
)

print(f"CV Scores: {scores}")
print(f"Mean R²: {scores.mean():.4f} (±{scores.std():.4f})")
```

## Feature Importance with Imputed Data

```python
from sklearn.ensemble import RandomForestRegressor

# Train Random Forest
pipeline = ImputedPipeline(
    imputer=KNNImputerMethod(k=5),
    model=RandomForestRegressor(n_estimators=100, random_state=42),
    scaler=False  # RF doesn't need scaling
)

pipeline.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': pipeline.model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance.head(10)['feature'],
         feature_importance.head(10)['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance (After Imputation)', fontweight='bold')
plt.tight_layout()
plt.show()
```

## Production Deployment

Save and load pipelines for production:

```python
import joblib

# Train final model
final_pipeline = ImputedPipeline(
    imputer=KNNImputerMethod(k=5),
    model=RandomForestRegressor(n_estimators=200, random_state=42),
    scaler=True
)

final_pipeline.fit(X_train, y_train)

# Save pipeline
joblib.dump(final_pipeline, 'production_model.pkl')
print("Model saved to production_model.pkl")

# Load and use in production
loaded_pipeline = joblib.load('production_model.pkl')
new_predictions = loaded_pipeline.predict(X_new_data)
```

## Handling New Missing Patterns

Monitor and adapt to new missingness patterns:

```python
class AdaptiveImputationPipeline:
    """Pipeline that adapts to changing missingness patterns."""

    def __init__(self, primary_imputer, fallback_imputer, model):
        self.primary_imputer = primary_imputer
        self.fallback_imputer = fallback_imputer
        self.model = model
        self.scaler = StandardScaler()
        self.training_missing_rate = None

    def fit(self, X, y):
        """Fit pipeline."""
        self.training_missing_rate = X.isna().sum().sum() / X.size

        X_imputed = self.primary_imputer.impute(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        """Predict with fallback logic."""
        current_missing_rate = X.isna().sum().sum() / X.size

        # If missingness pattern changes significantly, use fallback
        if abs(current_missing_rate - self.training_missing_rate) > 0.1:
            print(f"⚠️  Missingness rate changed significantly: "
                  f"{current_missing_rate:.2%} vs {self.training_missing_rate:.2%}")
            X_imputed = self.fallback_imputer.impute(X)
        else:
            X_imputed = self.primary_imputer.impute(X)

        X_scaled = self.scaler.transform(X_imputed)
        return self.model.predict(X_scaled)

# Usage
adaptive_pipeline = AdaptiveImputationPipeline(
    primary_imputer=KNNImputerMethod(k=5),
    fallback_imputer=MeanImputer(),
    model=Ridge(alpha=1.0)
)

adaptive_pipeline.fit(X_train, y_train)
predictions = adaptive_pipeline.predict(X_test)
```

## Key Takeaways

1. **Always split before imputing** to avoid data leakage
2. **Compare multiple imputation methods** for your specific task
3. **Use pipelines** for reproducibility and deployment
4. **Monitor performance** and adapt to changing data
5. **Cross-validate properly** with imputation inside CV loop
6. **Document your choices** for maintainability

## Complete Example Code

Run the full example from the repository root (it needs the `viz` extra):

```bash
poetry run python examples/ml_pipeline_example.py
```

This will generate (plots are saved in `examples/`):
- Performance comparison across methods
- Visualization of results
- Predictions vs actual values plot

## Next Steps

- Explore [Time Series Example](time-series.md) for temporal data
- Learn about [Custom Imputers](custom-imputers.md) for domain-specific logic
- Review [Best Practices](../user-guide/best-practices.md) for production deployment
- Check [Evaluation Metrics](../user-guide/evaluation.md) for assessing quality
