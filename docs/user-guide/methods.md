# Imputation Methods

This guide describes the core imputation methods available in the library, organized by category. The library exports 42 imputers in total; the remaining ones are listed under [Other Methods](#other-methods), and the [API Reference](../api/index.md) documents every class and constructor argument.

## Statistical Methods

Simple, fast methods based on basic statistics.

### Mean Imputation

Replace missing values with the column mean.

```python
from imputation_methods import MeanImputer

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
from imputation_methods import MeanImputer

df = pd.DataFrame({'A': [1, 2, np.nan, 4, 5]})
imputer = MeanImputer()
result = imputer.impute(df)
# Missing value imputed as: (1+2+4+5)/4 = 3.0
```

### Median Imputation

Replace missing values with the column median.

```python
from imputation_methods import MedianImputer

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
from imputation_methods import LOCFImputer

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
from imputation_methods import NOCBImputer

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

Impute using the average of the k nearest neighbors (scikit-learn's `KNNImputer`).

```python
from imputation_methods import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `n_neighbors`: Number of neighbors (default: 5)

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
- Requires tuning `n_neighbors`

**Hyperparameter tuning:**
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# Try different numbers of neighbors
for n_neighbors in [3, 5, 7, 10]:
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_imputed = imputer.impute(X_train)
    score = cross_val_score(Ridge(), X_imputed, y_train, cv=5).mean()
    print(f"n_neighbors={n_neighbors}: CV Score = {score:.4f}")
```

### Hot Deck Imputation

Randomly sample from similar observations ("donors").

```python
from imputation_methods import HotDeckImputer

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
- `stratify_cols`: Columns used to define similarity groups (must be numeric, e.g. integer-coded categories)
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
from imputation_methods import RegressionImputer

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
from imputation_methods import StochasticRegressionImputer

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
from imputation_methods import PMMImputer

imputer = PMMImputer(n_neighbors=5, random_state=42)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `n_neighbors`: Number of donor candidates (default: 5)
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
from imputation_methods import MICEImputer

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
- Flexible in principle (this implementation uses `BayesianRidge` for every column)
- Widely accepted in statistics

**Cons:**
- Computationally intensive
- May not converge
- Only `random_state` is configurable

**Note:** Full MICE involves creating multiple imputed datasets. This implementation returns a single imputation from scikit-learn's `IterativeImputer`.

## Tree-Based Methods

Methods using decision tree ensembles.

### MissForest

Random forest-based iterative imputation (scikit-learn's `IterativeImputer` with a `RandomForestRegressor` per column).

```python
from imputation_methods import MissForestImputer

imputer = MissForestImputer(random_state=42)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `random_state`: Random seed

**When to use:**
- Non-linear relationships
- Integer-coded categorical features mixed with continuous ones (all columns must be numeric)
- High-dimensional data

**Pros:**
- Handles non-linearity
- Can capture complex interactions
- Often very accurate

**Cons:**
- Slow for large datasets
- Computationally expensive
- Forest settings are fixed (only `random_state` is configurable)

## Matrix Completion Methods

Methods based on low-rank matrix approximation.

### SoftImpute

Low-rank matrix completion via nuclear-norm regularization.

```python
from imputation_methods import SoftImputeImputer

imputer = SoftImputeImputer(
    max_iter=100,
    init_fill_method='zero'
)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `max_iter`: Maximum iterations (default: 100)
- `init_fill_method`: Initialization strategy (`'zero'`, `'mean'`, `'median'` or `'min'`; default: `'zero'`)
- `shrinkage_value`: Amount subtracted from each singular value (default: 1/50 of the largest singular value of the initial fill)
- `convergence_threshold`: Relative change of the imputed entries at which to stop (default: 1e-3)

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

Probabilistic PCA for missing data imputation. Columns are standardized and a PPCA model is fitted by EM while the missing entries are repeatedly replaced by their expected values. Despite the name, this is maximum-likelihood PPCA (Tipping & Bishop, 1999): no priors are placed on the loadings.

```python
from imputation_methods import PPCAImputer

imputer = PPCAImputer(
    n_components=2,
    min_obs=1
)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `n_components`: Number of latent dimensions (default: 1; capped at the number of columns minus one)
- `min_obs`: Minimum observed values a column needs to take part in the model; other columns are mean-imputed
- `max_iter`: Maximum EM iterations (default: 500)
- `tol`: Convergence tolerance (default: 1e-6)

**When to use:**
- Data lies in low-dimensional subspace
- Want probabilistic interpretation
- High-dimensional data

**Pros:**
- Principled probabilistic approach
- Dimension reduction + imputation
- Explicit noise model

**Cons:**
- Requires choosing number of components
- Assumes Gaussian distribution
- Can be unstable

## Deep Learning Methods

Neural network-based approaches.

### Autoencoder Imputation

Use a neural network autoencoder (scikit-learn's `MLPRegressor` trained to reconstruct the mean-filled data) to predict missing values.

```python
from imputation_methods import AutoencoderImputer

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

The GAN-based approach of Yoon et al. (2018). A generator fills in missing entries while a discriminator tries to tell observed entries from imputed ones; a hint vector reveals part of the missingness mask to the discriminator, and a reconstruction loss keeps the generator faithful to the observed values. Implemented on NumPy, with the reference implementation's defaults.

```python
from imputation_methods import GAINImputer

imputer = GAINImputer(max_iter=2000, random_state=42)
df_imputed = imputer.impute(df)
```

**Parameters:**
- `batch_size`: Rows per training step (default: 128)
- `hint_rate`: Probability of revealing each mask entry to the discriminator (default: 0.9)
- `alpha`: Weight of the reconstruction loss (default: 100)
- `max_iter`: Training steps (default: 10000)
- `learning_rate`: Adam learning rate (default: 0.001)
- `random_state`: Random seed

**When to use:**
- Larger datasets with complex, non-linear dependencies between columns
- As a generative alternative to compare against MICE or MissForest

**Pros:**
- Learns the joint distribution without parametric assumptions
- Handles any missingness pattern across columns

**Cons:**
- Needs a reasonable amount of data; on small tables MICE is usually more accurate
- Training takes seconds rather than milliseconds
- Results depend on the random seed and training budget

## Advanced Statistical Methods

Sophisticated statistical approaches.

### Gaussian Process Imputation

Predict each column from the other columns with Gaussian process regression; missing entries are filled with the GP predictive mean.

```python
from imputation_methods import GaussianProcessImputer
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
- Smooth non-linear relationships between columns
- Small to moderate datasets

**Pros:**
- Flexible kernel choice
- Captures smooth non-linear relationships
- Principled probabilistic approach (the imputer returns point predictions only)

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
| GAIN | ⚡ | ⭐⭐⭐ | Complex | Large datasets, non-linear dependencies |
| Gaussian Process | ⚡ | ⭐⭐⭐⭐ | Complex | Smooth non-linear data |

## Other Methods

The library also provides these imputers (see the [API Reference](../api/index.md) for their arguments):

- **Statistical:** `ModeImputer`, `ConstantImputer`, `QuantileImputer`, `TrimmedMeanImputer`, `EndOfDistributionImputer`, `GroupMeanImputer`, `IndicatorImputer`
- **Sampling:** `RandomSamplingImputer`, `ColdDeckImputer`
- **Time series:** `ForwardFillFallbackImputer`, `InterpolationImputer`, `MovingAverageImputer`, `WeightedMovingAverageImputer`, `LinearTrendImputer`, `PolynomialTrendImputer`, `SeasonalImputer`, `KalmanFilterImputer`
- **Distance-based:** `RadiusNeighborsImputer`, `LocalMeanImputer`
- **Regression-based:** `BayesianRidgeImputer`, `HuberImputer`, `RANSACImputer`
- **Iterative:** `EMImputer` — iterative chained-equations imputation (`IterativeImputer`) with a configurable `max_iter` and `tol`; it is not closed-form EM for a multivariate normal
- **Ensemble:** `HybridImputer` (tries imputers in order until no NaNs remain), `StackingImputer` (element-wise mean or median of several imputers' outputs; `meta_strategy="weighted"` currently equals `"mean"`), `BaggingImputer` (bootstrap aggregating: averages a base imputer's imputations over resampled rows)

## Next Steps

- Learn about [Method Selection](selection.md) to choose the right approach
- Understand [Evaluation Metrics](evaluation.md) to assess quality
- Review [Best Practices](best-practices.md) for production use
- See [Examples](../examples/time-series.md) for real-world applications
