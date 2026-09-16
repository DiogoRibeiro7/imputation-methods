# Custom Imputers

Learn how to create your own custom imputation methods tailored to your specific domain and data characteristics.

## Overview

While the library provides 42 built-in imputers, sometimes you need custom logic for:

- Domain-specific imputation rules
- Combining multiple strategies
- Conditional logic based on data patterns
- Industry-specific requirements
- Novel imputation approaches

## Full Example

For complete, runnable code, see [`examples/custom_imputer_example.py`](https://github.com/DiogoRibeiro7/imputation-methods/blob/main/examples/custom_imputer_example.py) in the repository.

!!! note "Built-in equivalents"
    Some examples below re-implement logic that the library already ships, to show the pattern: see `ModeImputer`, `TrimmedMeanImputer`, `GroupMeanImputer` and `SeasonalImputer`. The `ModeImputer` and `HybridImputer` classes defined on this page are local examples that share their names with built-in classes (the built-in `HybridImputer` is a fallback chain of imputers), so don't mix them with `from imputation_methods import *`.

## Basic Custom Imputer

All custom imputers inherit from `BaseImputer`:

```python
from imputation_methods import BaseImputer
import pandas as pd

class MyCustomImputer(BaseImputer):
    """Template for creating custom imputers."""

    def __init__(self, param1: float = 1.0):
        """Initialize with any parameters you need."""
        self.param1 = param1

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implement your imputation logic.

        Args:
            df: DataFrame with missing values

        Returns:
            DataFrame with imputed values
        """
        # Validate input (required)
        df = self._ensure_numeric(df)

        # Create copy to avoid modifying original
        result = df.copy()

        # Your imputation logic here
        # ...

        return result
```

## Example 1: Mode Imputer

Impute using the most frequent value (mode).

```python
from imputation_methods import BaseImputer
import pandas as pd

class ModeImputer(BaseImputer):
    """Impute missing values using the mode (most frequent value).

    Useful for discrete or categorical data encoded as numbers.
    """

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            mode_value = result[column].mode()

            if len(mode_value) > 0:
                # Use first mode if multiple exist
                result[column] = result[column].fillna(mode_value[0])
            else:
                # Fallback to mean if mode can't be computed
                result[column] = result[column].fillna(result[column].mean())

        return result

# Usage
imputer = ModeImputer()
df_imputed = imputer.impute(df)
```

**Use cases:**
- Discrete numerical data (rating scores: 1-5)
- Categorical data encoded as integers
- Survey responses
- Ordinal data

## Example 2: Conditional Imputer

Use different strategies based on missingness percentage.

```python
from imputation_methods import BaseImputer
import pandas as pd

class ConditionalImputer(BaseImputer):
    """
    Impute using different strategies based on missingness percentage.

    - Low missingness (<10%): Use mean
    - Medium missingness (10-30%): Use median
    - High missingness (>30%): Use constant value

    Args:
        low_threshold: Threshold for low missingness (default: 0.1)
        high_threshold: Threshold for high missingness (default: 0.3)
        constant_value: Value to use for high missingness (default: 0)
    """

    def __init__(self, low_threshold=0.1, high_threshold=0.3, constant_value=0):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.constant_value = constant_value

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            missing_rate = result[column].isna().sum() / len(result)

            if missing_rate < self.low_threshold:
                # Low missingness: use mean
                fill_value = result[column].mean()
            elif missing_rate < self.high_threshold:
                # Medium missingness: use median
                fill_value = result[column].median()
            else:
                # High missingness: use constant (or consider dropping column)
                fill_value = self.constant_value

            result[column] = result[column].fillna(fill_value)

        return result

# Usage
imputer = ConditionalImputer(
    low_threshold=0.1,
    high_threshold=0.3,
    constant_value=0
)
df_imputed = imputer.impute(df)
```

**Use cases:**
- Datasets with varying missingness across columns
- When different strategies work better for different missingness levels
- Flagging high-missingness columns

## Example 3: Robust Imputer

Use trimmed mean to avoid outlier influence.

```python
from imputation_methods import BaseImputer
from scipy import stats
import pandas as pd

class RobustImputer(BaseImputer):
    """
    Impute using trimmed mean to avoid outlier influence.

    Uses the trimmed mean (mean after removing extreme values) which
    is more robust to outliers than standard mean.

    Args:
        trim_proportion: Proportion of values to trim from each end (default: 0.1)
    """

    def __init__(self, trim_proportion=0.1):
        self.trim_proportion = trim_proportion

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            # Calculate trimmed mean (removes top/bottom 10%)
            trimmed_mean = stats.trim_mean(
                result[column].dropna(),
                self.trim_proportion
            )
            result[column] = result[column].fillna(trimmed_mean)

        return result

# Usage
imputer = RobustImputer(trim_proportion=0.1)
df_imputed = imputer.impute(df)
```

**Use cases:**
- Data with outliers
- When mean is too sensitive
- Financial data with extreme values

## Example 4: Hybrid Imputer

Combine multiple strategies based on data characteristics.

```python
from imputation_methods import BaseImputer
import pandas as pd

class HybridImputer(BaseImputer):
    """
    Combine multiple imputation strategies.

    Uses different methods for different columns based on their
    statistical properties (e.g., skewness).

    Args:
        skewness_threshold: Threshold for determining skewed distributions (default: 1.0)
    """

    def __init__(self, skewness_threshold=1.0):
        self.skewness_threshold = skewness_threshold

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            # Calculate skewness
            skewness = result[column].skew()

            if abs(skewness) > self.skewness_threshold:
                # Skewed distribution: use median (more robust)
                fill_value = result[column].median()
                strategy = "median"
            else:
                # Normal-ish distribution: use mean
                fill_value = result[column].mean()
                strategy = "mean"

            result[column] = result[column].fillna(fill_value)
            print(f"   {column}: skewness={skewness:.2f} -> using {strategy}")

        return result

# Usage
imputer = HybridImputer(skewness_threshold=1.0)
df_imputed = imputer.impute(df)
```

**Use cases:**
- Mixed distribution types in same dataset
- Automated feature engineering pipelines
- When columns have very different characteristics

## Example 5: Group-Based Imputer

Impute within groups (e.g., by category or segment).

```python
from imputation_methods import BaseImputer
import pandas as pd

class GroupImputer(BaseImputer):
    """
    Impute based on groups.

    Fills missing values using statistics from the same group.

    Args:
        group_col: Column name to group by
    """

    def __init__(self, group_col):
        self.group_col = group_col

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        result = df.copy()

        # Group by the specified column and fill within groups
        for group_value in result[self.group_col].unique():
            if pd.notna(group_value):
                mask = result[self.group_col] == group_value

                for column in result.columns:
                    if column != self.group_col:
                        # Compute group mean
                        group_mean = result.loc[mask, column].mean()
                        # Fill missing values within group
                        result.loc[mask, column] = result.loc[mask, column].fillna(group_mean)

        return result

# Usage
# Impute separately for each category (e.g., department, region, product type)
imputer = GroupImputer(group_col='category')
df_imputed = imputer.impute(df)
```

**Use cases:**
- Hierarchical data (e.g., by department, region)
- When groups have different patterns
- Segmented analysis

## Example 6: Domain-Specific Imputer

Custom logic based on domain knowledge.

```python
from imputation_methods import BaseImputer
import pandas as pd
import numpy as np

class TemperatureImputer(BaseImputer):
    """
    Domain-specific imputer for temperature data.

    Uses physical constraints and seasonal patterns.

    Args:
        min_temp: Minimum physically plausible temperature
        max_temp: Maximum physically plausible temperature
        seasonal_period: Period for seasonal averaging (e.g., 24 for hourly data)
    """

    def __init__(self, min_temp=-50, max_temp=60, seasonal_period=24):
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.seasonal_period = seasonal_period

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            # Use seasonal average for imputation
            for i in range(len(result)):
                if pd.isna(result.loc[i, column]):
                    # Find values at same time of day/week
                    same_time_indices = result.index[
                        result.index % self.seasonal_period == i % self.seasonal_period
                    ]
                    same_time_values = result.loc[same_time_indices, column].dropna()

                    if len(same_time_values) > 0:
                        # Use median of same times
                        imputed_value = same_time_values.median()

                        # Apply physical constraints
                        imputed_value = np.clip(imputed_value, self.min_temp, self.max_temp)

                        result.loc[i, column] = imputed_value
                    else:
                        # Fallback to overall median
                        result.loc[i, column] = result[column].median()

        return result

# Usage
imputer = TemperatureImputer(
    min_temp=-40,
    max_temp=50,
    seasonal_period=24  # Hourly data with daily seasonality
)
df_imputed = imputer.impute(df)
```

**Use cases:**
- Any domain with physical constraints (temperature, pressure, concentration)
- Sensor data with known bounds
- Financial data with regulatory limits

## Advanced: Combining with Existing Imputers

Use built-in imputers within your custom logic:

```python
from imputation_methods import BaseImputer, MeanImputer, KNNImputer
import pandas as pd

class SmartImputer(BaseImputer):
    """
    Use KNN for low missingness, fallback to mean for high missingness.
    """

    def __init__(self, threshold=0.3, k=5):
        self.threshold = threshold
        self.knn_imputer = KNNImputer(n_neighbors=k)
        self.mean_imputer = MeanImputer()

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)

        missing_rate = df.isna().sum().sum() / df.size

        if missing_rate < self.threshold:
            # Low missingness: use accurate but slow KNN
            return self.knn_imputer.impute(df)
        else:
            # High missingness: use fast mean imputation
            return self.mean_imputer.impute(df)

# Usage
imputer = SmartImputer(threshold=0.3, k=5)
df_imputed = imputer.impute(df)
```

## Testing Custom Imputers

Always test your custom imputers:

```python
import unittest
import pandas as pd
import numpy as np

class TestModeImputer(unittest.TestCase):
    """Test custom mode imputer."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({
            'discrete': [1, 2, 2, np.nan, 2, 3, np.nan]
        })

    def test_imputes_with_mode(self):
        """Test that mode imputer uses most frequent value."""
        imputer = ModeImputer()
        result = imputer.impute(self.df)

        # Mode is 2 (appears 3 times)
        self.assertEqual(result['discrete'].iloc[3], 2.0)
        self.assertEqual(result['discrete'].iloc[6], 2.0)

    def test_no_missing_after_imputation(self):
        """Test that all NaN values are filled."""
        imputer = ModeImputer()
        result = imputer.impute(self.df)

        self.assertFalse(result.isna().any().any())

    def test_preserves_observed_values(self):
        """Test that observed values are unchanged."""
        imputer = ModeImputer()
        result = imputer.impute(self.df)

        observed_mask = self.df['discrete'].notna()
        pd.testing.assert_series_equal(
            self.df.loc[observed_mask, 'discrete'],
            result.loc[observed_mask, 'discrete']
        )

if __name__ == '__main__':
    unittest.main()
```

## Integration with Pipelines

Use custom imputers in scikit-learn pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

class ImputationTransformer:
    """Wrapper for sklearn compatibility."""

    def __init__(self, imputer):
        self.imputer = imputer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.imputer.impute(X)

# Create pipeline with custom imputer
pipeline = Pipeline([
    ('impute', ImputationTransformer(HybridImputer())),
    ('scale', StandardScaler()),
    ('model', RandomForestRegressor())
])

# Use like any sklearn model
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## Best Practices for Custom Imputers

### 1. Always Call `_ensure_numeric()`

```python
def impute(self, df: pd.DataFrame) -> pd.DataFrame:
    df = self._ensure_numeric(df)  # Validates input
    # Your logic here...
```

### 2. Work on a Copy

```python
def impute(self, df: pd.DataFrame) -> pd.DataFrame:
    df = self._ensure_numeric(df)
    result = df.copy()  # Don't modify original
    # Your logic here...
    return result
```

### 3. Document Thoroughly

```python
class MyImputer(BaseImputer):
    """
    One-line summary.

    Detailed explanation of what this imputer does,
    when to use it, and any important considerations.

    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2

    Examples:
        >>> imputer = MyImputer(param1=5)
        >>> df_imputed = imputer.impute(df)
    """
```

### 4. Handle Edge Cases

```python
def impute(self, df: pd.DataFrame) -> pd.DataFrame:
    df = self._ensure_numeric(df)
    result = df.copy()

    for column in result.columns:
        # Check if column has any non-missing values
        if result[column].notna().sum() == 0:
            # All values missing - handle appropriately
            result[column] = 0  # or raise error, or skip

        # Your imputation logic...

    return result
```

### 5. Add Logging (Optional)

```python
import logging

class MyImputer(BaseImputer):
    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger = logging.getLogger(__name__)
        logger.info(f"Imputing {df.shape[0]} rows, {df.shape[1]} columns")

        df = self._ensure_numeric(df)
        result = df.copy()

        # Your logic...

        logger.info(f"Imputation complete: {result.isna().sum().sum()} remaining NaN")
        return result
```

## Complete Example

Run the full example from the repository root to see the custom imputers in action (it needs the `viz` extra):

```bash
poetry run python examples/custom_imputer_example.py
```

This will demonstrate:
- Examples 1–5 from this page
- Performance comparison
- Visualization of results
- Use case recommendations

## Key Takeaways

1. **Inherit from `BaseImputer`** for consistent interface
2. **Call `_ensure_numeric()`** to validate input
3. **Work on copies** to avoid side effects
4. **Document thoroughly** for maintainability
5. **Test comprehensively** before production use
6. **Consider edge cases** (all missing, no missing, etc.)

## When to Create Custom Imputers

Create a custom imputer when:

- ✅ You have domain-specific knowledge
- ✅ Standard methods don't handle your data well
- ✅ You need conditional logic based on data properties
- ✅ You want to combine multiple strategies
- ✅ You have physical/business constraints to enforce

Don't create a custom imputer when:

- ❌ A built-in method already does what you need
- ❌ You're just wrapping an existing method without adding value
- ❌ The logic is overly complex and hard to maintain
- ❌ You haven't validated it's actually better than standard methods

## Next Steps

- Review [Methods Guide](../user-guide/methods.md) for built-in options
- Check [Best Practices](../user-guide/best-practices.md) for production deployment
- See [Time Series Example](time-series.md) and [ML Pipeline Example](ml-pipeline.md)
- Consult [API Reference](../api/index.md) for technical details
