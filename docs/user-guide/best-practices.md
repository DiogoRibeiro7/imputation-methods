# Best Practices

Production-ready guidelines for implementing robust imputation pipelines.

## General Principles

### 1. Understand Before Imputing

Always analyze your missing data before choosing an imputation strategy.

```python
import pandas as pd
import missingno as msno  # Separate package: pip install missingno
import matplotlib.pyplot as plt

def analyze_missingness(df):
    """Comprehensive missingness analysis."""

    print("="*80)
    print("MISSINGNESS ANALYSIS")
    print("="*80)

    # Overall statistics
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    print(f"\nTotal cells: {total_cells:,}")
    print(f"Missing cells: {missing_cells:,} ({missing_cells/total_cells*100:.2f}%)")

    # Per-column statistics
    print("\nMissing values per column:")
    missing_by_col = df.isna().sum().sort_values(ascending=False)
    for col, count in missing_by_col[missing_by_col > 0].items():
        pct = count / len(df) * 100
        print(f"  {col:20s}: {count:6d} ({pct:5.2f}%)")

    # Visualize patterns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Matrix plot
    msno.matrix(df, ax=axes[0])
    axes[0].set_title('Missingness Pattern Matrix')

    # Bar plot
    msno.bar(df, ax=axes[1])
    axes[1].set_title('Missing Data by Column')

    plt.tight_layout()
    plt.show()

    # Test for MCAR using Little's test (simplified)
    return df

# Usage
df = analyze_missingness(your_data)
```

### 2. Document Your Decisions

Always document why you chose a particular imputation method.

```python
# Good practice: Create imputation config
IMPUTATION_CONFIG = {
    'version': '1.0',
    'date': '2024-11-20',
    'method': 'KNNImputerMethod',
    'parameters': {
        'k': 5
    },
    'rationale': """
        Chose KNN with k=5 because:
        1. Features show moderate correlation (correlation matrix analysis)
        2. Missing rate is 15% (appropriate for KNN)
        3. Data is MAR based on missingness analysis
        4. KNN outperformed mean/median in CV (R²: 0.85 vs 0.78)
    """,
    'validation': {
        'cv_score': 0.85,
        'test_rmse': 2.34,
        'test_mae': 1.87
    },
    'assumptions': [
        'Features are scaled before imputation',
        'Missing data mechanism is MAR',
        'Correlations between features are stable'
    ],
    'limitations': [
        'Computationally expensive for >100K rows',
        'Sensitive to outliers',
        'Requires feature scaling'
    ]
}

# Save configuration
import json
with open('imputation_config.json', 'w') as f:
    json.dump(IMPUTATION_CONFIG, f, indent=2)
```

## Data Pipeline Best Practices

### 1. Split Before Imputing

**❌ INCORRECT: Impute before splitting**
```python
# This causes data leakage!
df_imputed = imputer.impute(df)
train, test = train_test_split(df_imputed)
```

**✅ CORRECT: Split before imputing**
```python
# Split first
train, test = train_test_split(df)

# Impute each split separately (impute() only sees the rows it is given)
imputer = KNNImputerMethod(k=5)
train_imputed = imputer.impute(train)
test_imputed = imputer.impute(test)
```

### 2. Create Reproducible Pipelines

Use scikit-learn pipelines for reproducibility:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Custom imputation transformer
class ImputationTransformer:
    """Scikit-learn compatible imputation transformer."""

    def __init__(self, imputer):
        self.imputer = imputer

    def fit(self, X, y=None):
        # Imputers in this library have no fitted state: impute() estimates
        # everything from the data passed to transform()
        return self

    def transform(self, X):
        return self.imputer.impute(X)

    def fit_transform(self, X, y=None):
        return self.transform(X)

# Create pipeline
from imputation_methods import KNNImputerMethod

pipeline = Pipeline([
    ('impute', ImputationTransformer(KNNImputerMethod(k=5))),
    ('scale', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

# Use pipeline
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

# Save entire pipeline
import joblib
joblib.dump(pipeline, 'model_pipeline.pkl')
```

### 3. Handle Categorical Variables

Encode categorical variables before imputation:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_for_imputation(df):
    """Prepare mixed-type dataframe for imputation."""

    df_processed = df.copy()

    # Separate numeric and categorical
    numeric_cols = df.select_dtypes(include='number').columns
    # 'string' also selects pandas 3's default str dtype
    categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle missing values in categorical columns
        df_processed[col] = df_processed[col].fillna('MISSING')
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

    # Now all columns are numeric and can be imputed
    return df_processed, label_encoders, numeric_cols, categorical_cols

# Usage
df_processed, encoders, num_cols, cat_cols = prepare_for_imputation(df)

# Impute
from imputation_methods import KNNImputerMethod
imputer = KNNImputerMethod(k=5)
df_imputed = imputer.impute(df_processed)

# Decode categorical variables back
for col in cat_cols:
    df_imputed[col] = encoders[col].inverse_transform(
        df_imputed[col].round().astype(int)
    )
```

## Production Deployment

### 1. Version Control Your Imputers

Track imputer versions and parameters:

```python
class VersionedImputer:
    """Imputer with version tracking."""

    def __init__(self, imputer, version, metadata=None):
        self.imputer = imputer
        self.version = version
        self.metadata = metadata or {}
        self.created_at = pd.Timestamp.now()

    def impute(self, df):
        """Impute with logging."""
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"Using imputer version {self.version}")
        logger.info(f"Input shape: {df.shape}")
        logger.info(f"Missing values: {df.isna().sum().sum()}")

        result = self.imputer.impute(df)

        logger.info(f"Imputation complete")
        logger.info(f"Output shape: {result.shape}")

        return result

    def save(self, path):
        """Save imputer with metadata."""
        import joblib

        data = {
            'imputer': self.imputer,
            'version': self.version,
            'metadata': self.metadata,
            'created_at': self.created_at
        }

        joblib.dump(data, path)

    @classmethod
    def load(cls, path):
        """Load versioned imputer."""
        import joblib
        data = joblib.load(path)

        return cls(
            imputer=data['imputer'],
            version=data['version'],
            metadata=data['metadata']
        )

# Usage
from imputation_methods import KNNImputerMethod

imputer = VersionedImputer(
    imputer=KNNImputerMethod(k=5),
    version='1.0.0',
    metadata={
        'trained_on': 'training_data_2024_11',
        'purpose': 'production_model_v2'
    }
)

imputer.save('imputer_v1.0.0.pkl')
```

### 2. Implement Fallback Strategies

Always have a fallback for when primary imputation fails:

```python
class RobustImputer:
    """Imputer with fallback strategies."""

    def __init__(self, primary_imputer, fallback_imputer=None, timeout=30.0):
        self.primary_imputer = primary_imputer
        self.fallback_imputer = fallback_imputer or MeanImputer()
        self.timeout = timeout

    def impute(self, df):
        """Impute with fallback and timeout."""
        import time
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Try primary imputer with timeout
            start = time.time()
            result = self.primary_imputer.impute(df)
            elapsed = time.time() - start

            if elapsed > self.timeout:
                logger.warning(
                    f"Primary imputer took {elapsed:.2f}s (timeout: {self.timeout}s)"
                )

            # Validation checks
            if result.isna().any().any():
                raise ValueError("Primary imputer did not fill all NaN values")

            if not result.shape == df.shape:
                raise ValueError("Shape mismatch after imputation")

            return result

        except Exception as e:
            logger.error(f"Primary imputer failed: {e}")
            logger.info("Falling back to simple imputer")

            try:
                return self.fallback_imputer.impute(df)
            except Exception as e2:
                logger.error(f"Fallback imputer also failed: {e2}")
                raise RuntimeError("All imputation strategies failed") from e2

# Usage
from imputation_methods import KNNImputerMethod, MeanImputer

robust_imputer = RobustImputer(
    primary_imputer=KNNImputerMethod(k=5),
    fallback_imputer=MeanImputer(),
    timeout=10.0
)

df_imputed = robust_imputer.impute(df)
```

### 3. Monitor in Production

Implement monitoring for production systems:

```python
class MonitoredImputer:
    """Imputer with production monitoring."""

    def __init__(self, imputer, alert_missing_rate=0.5, alert_email=None):
        self.imputer = imputer
        self.alert_missing_rate = alert_missing_rate
        self.alert_email = alert_email
        self.stats_history = []

    def impute(self, df):
        """Impute with monitoring."""
        import time
        import logging

        logger = logging.getLogger(__name__)
        start_time = time.time()

        # Pre-imputation statistics
        missing_rate = df.isna().sum().sum() / df.size
        missing_by_col = df.isna().sum()

        # Alert if missing rate is too high
        if missing_rate > self.alert_missing_rate:
            self.send_alert(
                f"High missing rate: {missing_rate:.2%} (threshold: {self.alert_missing_rate:.2%})"
            )

        # Impute
        result = self.imputer.impute(df)
        elapsed = time.time() - start_time

        # Post-imputation statistics
        stats = {
            'timestamp': pd.Timestamp.now(),
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'missing_rate': missing_rate,
            'missing_by_col': missing_by_col.to_dict(),
            'imputation_time': elapsed,
            'mean_before': df.mean().mean(),
            'mean_after': result.mean().mean(),
            'std_before': df.std().mean(),
            'std_after': result.std().mean()
        }

        self.stats_history.append(stats)
        logger.info(f"Imputation complete: {elapsed:.2f}s, missing_rate: {missing_rate:.2%}")

        return result

    def send_alert(self, message):
        """Send alert (implement your notification logic)."""
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"⚠️  ALERT: {message}")

        # Implement email/SMS/Slack notification here
        if self.alert_email:
            # send_email(self.alert_email, "Imputation Alert", message)
            pass

    def get_stats_summary(self):
        """Get summary statistics."""
        if not self.stats_history:
            return "No statistics available"

        df = pd.DataFrame(self.stats_history)
        return df.describe()

# Usage
monitored_imputer = MonitoredImputer(
    imputer=KNNImputerMethod(k=5),
    alert_missing_rate=0.4,
    alert_email="data-team@example.com"
)

# In production loop
for batch in data_stream:
    imputed = monitored_imputer.impute(batch)
    # Process imputed data...

# Review statistics
print(monitored_imputer.get_stats_summary())
```

## Performance Optimization

### 1. Scale Features Before KNN

KNN is sensitive to feature scales:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imputation_methods import KNNImputerMethod

# Scale before imputation. StandardScaler ignores NaNs when fitting
# and keeps them in the output, so it only uses observed values.
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df), columns=df.columns, index=df.index
)

# Now impute
imputer = KNNImputerMethod(k=5)
df_imputed = imputer.impute(df_scaled)

# Scale back if needed
df_imputed = pd.DataFrame(
    scaler.inverse_transform(df_imputed), columns=df.columns, index=df.index
)
```

### 2. Sample for Large Datasets

For very large datasets, consider imputing smaller subsets of rows at a time:

```python
def smart_impute_large_dataset(df, imputer, sample_size=10000):
    """Impute large dataset in row subsets of at most sample_size rows."""

    if len(df) <= sample_size:
        # Small enough, impute directly
        return imputer.impute(df)

    # For large datasets, use simple imputation
    # or a subset-based strategy
    from imputation_methods import MeanImputer

    if len(df) > 1000000:
        # Very large: use fast method
        return MeanImputer().impute(df)
    else:
        # Large: impute consecutive chunks. Imputers keep no fitted state,
        # so each chunk is imputed from its own rows only.
        chunks = [
            imputer.impute(df.iloc[start:start + sample_size])
            for start in range(0, len(df), sample_size)
        ]
        return pd.concat(chunks)

# Usage
df_imputed = smart_impute_large_dataset(large_df, KNNImputerMethod(k=5))
```

### 3. Parallelize When Possible

For independent features, consider parallel imputation:

```python
from joblib import Parallel, delayed
import pandas as pd

def impute_column(df, col, imputer):
    """Impute single column."""
    df_col = df[[col]]
    return imputer.impute(df_col)[col]

def parallel_impute(df, imputer, n_jobs=-1):
    """Impute columns in parallel."""

    results = Parallel(n_jobs=n_jobs)(
        delayed(impute_column)(df, col, imputer)
        for col in df.columns
    )

    return pd.DataFrame({col: results[i]
                        for i, col in enumerate(df.columns)},
                       index=df.index)

# Usage (only works for column-independent methods)
from imputation_methods import MedianImputer
df_imputed = parallel_impute(df, MedianImputer(), n_jobs=4)
```

## Testing Best Practices

### 1. Unit Tests for Imputation Logic

```python
import unittest
import pandas as pd
import numpy as np
from imputation_methods import MeanImputer, KNNImputerMethod

class TestImputationPipeline(unittest.TestCase):
    """Test imputation logic."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [5, np.nan, 7, 8, 9]
        })

    def test_mean_imputer_fills_all_nan(self):
        """Test that mean imputer fills all NaN values."""
        imputer = MeanImputer()
        result = imputer.impute(self.df)
        self.assertFalse(result.isna().any().any())

    def test_imputation_preserves_shape(self):
        """Test that imputation preserves DataFrame shape."""
        imputer = KNNImputerMethod(k=2)
        result = imputer.impute(self.df)
        self.assertEqual(result.shape, self.df.shape)

    def test_imputation_preserves_observed_values(self):
        """Test that observed values are not changed."""
        imputer = MeanImputer()
        result = imputer.impute(self.df)

        # Check observed values unchanged
        observed_mask = ~self.df.isna()
        pd.testing.assert_frame_equal(
            self.df[observed_mask],
            result[observed_mask]
        )

    def test_imputer_handles_no_missing(self):
        """Test imputer works when no values are missing."""
        df_complete = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        imputer = MeanImputer()
        result = imputer.impute(df_complete)
        pd.testing.assert_frame_equal(df_complete, result)

    def test_imputer_handles_all_missing(self):
        """Test imputer handles column with all missing values."""
        df = pd.DataFrame({'A': [np.nan, np.nan, np.nan]})
        imputer = MeanImputer()
        result = imputer.impute(df)
        # Mean of all NaN is NaN - this is expected behavior
        self.assertTrue(result.isna().all().all())

if __name__ == '__main__':
    unittest.main()
```

### 2. Integration Tests

```python
def test_end_to_end_pipeline():
    """Test complete pipeline from raw data to predictions."""

    # Load data
    df = pd.read_csv('test_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Impute
    from imputation_methods import KNNImputerMethod
    imputer = KNNImputerMethod(k=5)
    X_train_imputed = imputer.impute(X_train)
    X_test_imputed = imputer.impute(X_test)

    # Train model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_imputed, y_train)

    # Predict
    predictions = model.predict(X_test_imputed)

    # Evaluate
    from sklearn.metrics import r2_score
    score = r2_score(y_test, predictions)

    # Assert minimum performance
    assert score > 0.7, f"Model performance too low: R²={score:.4f}"

    print(f"✓ End-to-end test passed: R²={score:.4f}")

# Run test
test_end_to_end_pipeline()
```

## Common Pitfalls to Avoid

### ❌ 1. Imputing Before Splitting

```python
# WRONG
df_imputed = imputer.impute(df)
train, test = train_test_split(df_imputed)  # Data leakage!
```

### ❌ 2. Not Handling Categorical Variables

```python
# WRONG
df_imputed = imputer.impute(df_with_categorical)  # Will raise error
```

### ❌ 3. Using Same Imputer on Different Scales

```python
# WRONG for KNN
imputer = KNNImputerMethod(k=5)
# Using same imputer on features with vastly different scales
```

### ❌ 4. Ignoring Computational Constraints

```python
# WRONG for large datasets
from imputation_methods import MissForestImputer

imputer = MissForestImputer()  # Too slow for >100K rows
df_imputed = imputer.impute(very_large_df)
```

### ❌ 5. Not Validating Imputation

```python
# WRONG
df_imputed = imputer.impute(df)
# No validation that imputation makes sense
```

## Checklist for Production

- [ ] Analyzed missingness patterns
- [ ] Documented method selection rationale
- [ ] Tested on representative data
- [ ] Implemented proper train/test split
- [ ] Added fallback strategies
- [ ] Implemented monitoring
- [ ] Created unit and integration tests
- [ ] Validated imputation quality
- [ ] Documented assumptions and limitations
- [ ] Set up alerting for anomalies
- [ ] Versioned imputer and configuration
- [ ] Logged imputation statistics
- [ ] Reviewed with domain experts

## Next Steps

- See [Examples](../examples/time-series.md) for complete implementations
- Review [Evaluation](evaluation.md) for quality assessment
- Check [Method Selection](selection.md) for choosing the right approach
- Consult [API Reference](../api/index.md) for technical details
