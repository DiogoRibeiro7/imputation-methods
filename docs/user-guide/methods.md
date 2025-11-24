# Imputation Methods

This comprehensive guide describes all 16+ imputation methods available in the library, organized by category.

## Statistical Methods

Simple, fast methods based on basic statistics.

### Mean Imputation

Replace missing values with the column mean.

```python
from imputation_showcase import MeanImputer

imputer = MeanImputer()
df_imputed = imputer.impute(df)
```

**When to use:**
- Quick baseline for comparison
- Data is approximately normally distributed
- Missing data is MCAR

**Pros:**
- Very fast (O(n))
- Easy to understand and implement
- No hyperparameters

**Cons:**
- Reduces variance
- Ignores relationships between variables
- Can distort distributions

**Example:**
```python
import pandas as pd
import numpy as np
from imputation_showcase import MeanImputer

df = pd.DataFrame({'A': [1, 2, np.nan, 4, 5]})
imputer = MeanImputer()
result = imputer.impute(df)
# Missing value imputed as: (1+2+4+5)/4 = 3.0
```

### Median Imputation

Replace missing values with the column median.

```python
from imputation_showcase import MedianImputer

imputer = MedianImputer()
df_imputed = imputer.impute(df)
```

**When to use:**
- Data has outliers
- Skewed distributions
- More robust alternative to mean

**Pros:**
- Robust to outliers
- Fast computation
- Simple interpretation

**Cons:**
- Same limitations as mean imputation
- Still ignores variable relationships

## Time Series Methods

Methods designed specifically for temporal data.

### LOCF (Last Observation Carried Forward)

Fill missing values with the most recent observed value.

```python
from imputation_showcase import LOCFImputer

imputer = LOCFImputer()
df_imputed = imputer.impute(df)
```

**When to use:**
- Time series data with temporal ordering
- Slowly changing variables
- Sensor data with intermittent failures

**Pros:**
- Preserves temporal structure
- Intuitive for time series
- Very fast

**Cons:**
- Introduces bias in long gaps
- Assumes persistence (values don't change)
- First value cannot be imputed if missing

**Example:**
```python
df = pd.DataFrame({'sensor': [20.5, np.nan, np.nan, 22.0, np.nan]})
# Result: [20.5, 20.5, 20.5, 22.0, 22.0]
```

### NOCB (Next Observation Carried Backward)

Fill missing values with the next observed value.

```python
from imputation_showcase import NOCBImputer

imputer = NOCBImputer()
df_imputed = imputer.impute(df)
```

**When to use:**
- Backward-looking analysis
- Complementary to LOCF
- Event-driven data

**Pros:**
- Useful for certain temporal patterns
- Fast computation
- Complementary to LOCF

**Cons:**
- Last value cannot be imputed if missing
- Less intuitive than forward filling
- Can introduce look-ahead bias

## Distance-Based Methods

Methods that use similarity between observations.

### K-Nearest Neighbors (KNN)

Impute using weighted average of k-nearest neighbors.

```python
from imputation_showcase import KNNImputerMethod

imputer = KNNImputerMethod(k=5)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `k`: Number of neighbors (default: 5)

**When to use:**
- Features are correlated
- Moderate-sized datasets
- MAR or MCAR data

**Pros:**
- Considers relationships between variables
- More accurate than simple methods
- Well-established algorithm

**Cons:**
- Computationally expensive for large datasets
- Sensitive to feature scaling
- Requires tuning k parameter

**Hyperparameter tuning:**
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# Try different k values
for k in [3, 5, 7, 10]:
    imputer = KNNImputerMethod(k=k)
    X_imputed = imputer.impute(X_train)
    score = cross_val_score(Ridge(), X_imputed, y_train, cv=5).mean()
    print(f"k={k}: CV Score = {score:.4f}")
```

### Hot Deck Imputation

Randomly sample from similar observations ("donors").

```python
from imputation_showcase import HotDeckImputer

# Simple random sampling
imputer = HotDeckImputer(random_state=42)

# Stratified by categorical variables
imputer = HotDeckImputer(
    stratify_cols=['category_A', 'category_B'],
    random_state=42
)

df_imputed = imputer.impute(df)
```

**Parameters:**
- `stratify_cols`: Columns used to define similarity groups
- `random_state`: Random seed for reproducibility

**When to use:**
- Preserving realistic value distributions
- Categorical stratification available
- Avoiding over-smoothing

**Pros:**
- Preserves empirical distribution
- Adds realistic variability
- Can stratify by categories

**Cons:**
- Random (results vary)
- Less accurate than model-based methods
- Requires sufficient donors

## Regression-Based Methods

Methods that predict missing values using regression models.

### Regression Imputation

Predict missing values using linear regression on other features.

```python
from imputation_showcase import RegressionImputer

imputer = RegressionImputer()
df_imputed = imputer.impute(df)
```

**When to use:**
- Linear relationships between features
- MAR data
- Need deterministic imputation

**Pros:**
- Uses variable relationships
- More accurate than mean/median
- Deterministic results

**Cons:**
- Underestimates variance
- Assumes linear relationships
- Can produce unrealistic values

### Stochastic Regression Imputation

Regression imputation with added noise to preserve variance.

```python
from imputation_showcase import StochasticRegressionImputer

imputer = StochasticRegressionImputer(random_state=42)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `random_state`: Random seed for noise generation

**When to use:**
- Need to preserve variance
- Statistical inference on imputed data
- Linear relationships exist

**Pros:**
- Preserves variance
- Uses feature relationships
- More realistic than deterministic regression

**Cons:**
- Introduces randomness
- Still assumes linearity
- Requires estimate of residual variance

### Predictive Mean Matching (PMM)

Regression-based method that selects actual observed values.

```python
from imputation_showcase import PMMImputer

imputer = PMMImputer(k=5, random_state=42)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `k`: Number of donor candidates
- `random_state`: Random seed

**When to use:**
- Want realistic values from actual data
- Avoid implausible predictions
- Preserve distributions

**Pros:**
- Imputes only observed values
- Preserves distributions
- Robust to model misspecification

**Cons:**
- More complex than simple regression
- Requires sufficient donors
- Computationally more expensive

### MICE (Multiple Imputation by Chained Equations)

Iteratively imputes each variable using the others as predictors.

```python
from imputation_showcase import MICEImputer

imputer = MICEImputer(random_state=42)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `random_state`: Random seed for reproducibility

**When to use:**
- Multiple variables have missing data
- Complex relationships between variables
- Need statistically principled approach

**Pros:**
- Handles multiple missing variables well
- Flexible (can use different models per variable)
- Widely accepted in statistics

**Cons:**
- Computationally intensive
- May not converge
- Requires careful tuning

**Note:** Full MICE involves creating multiple imputed datasets. This implementation returns a single imputation.

## Tree-Based Methods

Methods using decision tree ensembles.

### MissForest

Random forest-based iterative imputation.

```python
from imputation_showcase import MissForestImputer

imputer = MissForestImputer(random_state=42)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `random_state`: Random seed

**When to use:**
- Non-linear relationships
- Mixed data types (numeric/categorical)
- High-dimensional data

**Pros:**
- Handles non-linearity
- Can capture complex interactions
- Often very accurate

**Cons:**
- Slow for large datasets
- Computationally expensive
- Many hyperparameters (inherited from RandomForest)

## Matrix Completion Methods

Methods based on low-rank matrix approximation.

### SoftImpute

Low-rank matrix completion via nuclear-norm regularization.

```python
from imputation_showcase import SoftImputeImputer

imputer = SoftImputeImputer(
    max_iters=100,
    init_fill_method='zero'
)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `max_iters`: Maximum iterations
- `init_fill_method`: Initialization strategy ('zero', 'mean')

**When to use:**
- Data has low-rank structure
- Collaborative filtering-type problems
- High correlation between columns

**Pros:**
- Exploits low-rank structure
- Good for high-dimensional data
- Theoretically well-founded

**Cons:**
- Assumes low-rank structure
- Computationally intensive
- May over-smooth

### Bayesian PCA

Probabilistic PCA for missing data imputation.

```python
from imputation_showcase import BayesianPCAImputer

imputer = BayesianPCAImputer(
    n_components=2,
    min_obs=1
)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `n_components`: Number of latent dimensions
- `min_obs`: Minimum observations required

**When to use:**
- Data lies in low-dimensional subspace
- Want probabilistic interpretation
- High-dimensional data

**Pros:**
- Principled probabilistic approach
- Dimension reduction + imputation
- Handles uncertainty

**Cons:**
- Requires choosing number of components
- Assumes Gaussian distribution
- Can be unstable

## Deep Learning Methods

Neural network-based approaches.

### Autoencoder Imputation

Use neural network autoencoder to reconstruct missing values.

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
- `hidden_layer_sizes`: Network architecture
- `max_iter`: Training iterations
- `random_state`: Random seed

**When to use:**
- Large datasets
- Complex non-linear patterns
- Computational resources available

**Pros:**
- Can learn complex patterns
- Flexible architecture
- Good for large data

**Cons:**
- Requires large datasets
- Many hyperparameters
- Slow training
- Less interpretable

### GAIN (Generative Adversarial Imputation Networks)

GAN-based approach for missing data (simplified implementation).

```python
from imputation_showcase import GAINImputer

imputer = GAINImputer(random_state=42)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `random_state`: Random seed

**When to use:**
- Very large datasets
- Complex missing patterns
- State-of-the-art accuracy needed

**Pros:**
- State-of-the-art performance
- Can handle complex patterns
- Preserves distributions well

**Cons:**
- Very complex
- Requires significant computational resources
- Difficult to tune
- Black box

**Note:** This is a simplified implementation using iterative imputation.

## Advanced Statistical Methods

Sophisticated statistical approaches.

### Gaussian Process Imputation

Use GP regression for spatially-correlated data.

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
- `kernel`: Covariance kernel (default: RBF)
- `alpha`: Noise level
- `random_state`: Random seed

**When to use:**
- Spatial/temporal correlations
- Small to moderate datasets
- Need uncertainty estimates

**Pros:**
- Provides uncertainty estimates
- Flexible kernel choice
- Principled probabilistic approach

**Cons:**
- Computationally expensive (O(n³))
- Requires kernel selection
- Doesn't scale to large datasets

## Quick Reference Table

| Method | Speed | Accuracy | Complexity | Best For |
|--------|-------|----------|------------|----------|
| Mean | ⚡⚡⚡⚡⚡ | ⭐ | Simple | Quick baseline |
| Median | ⚡⚡⚡⚡⚡ | ⭐ | Simple | Outlier-robust baseline |
| LOCF | ⚡⚡⚡⚡⚡ | ⭐⭐ | Simple | Time series |
| NOCB | ⚡⚡⚡⚡⚡ | ⭐⭐ | Simple | Time series |
| KNN | ⚡⚡⚡ | ⭐⭐⭐ | Moderate | General purpose |
| Hot Deck | ⚡⚡⚡⚡ | ⭐⭐ | Simple | Preserve distributions |
| Regression | ⚡⚡⚡ | ⭐⭐⭐ | Moderate | Linear relationships |
| Stochastic Reg. | ⚡⚡⚡ | ⭐⭐⭐ | Moderate | Preserve variance |
| PMM | ⚡⚡ | ⭐⭐⭐⭐ | Moderate | Realistic values |
| MICE | ⚡⚡ | ⭐⭐⭐⭐ | Complex | Multiple variables |
| MissForest | ⚡ | ⭐⭐⭐⭐ | Complex | Non-linear patterns |
| SoftImpute | ⚡⚡ | ⭐⭐⭐ | Moderate | Low-rank data |
| Bayesian PCA | ⚡⚡ | ⭐⭐⭐ | Moderate | High-dimensional |
| Autoencoder | ⚡ | ⭐⭐⭐⭐ | Complex | Large datasets |
| GAIN | ⚡ | ⭐⭐⭐⭐⭐ | Very Complex | State-of-the-art |
| Gaussian Process | ⚡ | ⭐⭐⭐⭐ | Complex | Spatial data |

## Next Steps

- Learn about [Method Selection](selection.md) to choose the right approach
- Understand [Evaluation Metrics](evaluation.md) to assess quality
- Review [Best Practices](best-practices.md) for production use
- See [Examples](../examples/time-series.md) for real-world applications
