# Imputation Showcase

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-black.svg)](https://flake8.pycqa.org/)
[![CI](https://github.com/DiogoRibeiro7/imputation-showcase/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/imputation-showcase/actions)

A comprehensive Python library showcasing **41 state-of-the-art imputation techniques** for handling missing data in numerical datasets. Built with pandas and scikit-learn, this library provides a unified interface for comparing and evaluating different imputation strategies.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Available Methods](#available-methods)
- [Usage Examples](#usage-examples)
- [Evaluation Metrics](#evaluation-metrics)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Features

- **41 imputation methods** ranging from simple statistical approaches to advanced machine learning and robust techniques
- **Unified API** with consistent interface across all imputation methods
- **Type-safe** implementation with proper type hints
- **Comprehensive testing** with pytest and coverage reporting
- **Production-ready** with CI/CD pipeline and code quality checks
- **Well-documented** with examples and docstrings
- **Evaluation utilities** for comparing imputation quality (RMSE, MAE)

## Installation

### Using Poetry (Recommended)

```bash
poetry add imputation-showcase
```

### Using pip

```bash
pip install imputation-showcase
```

### From Source

```bash
git clone https://github.com/DiogoRibeiro7/imputation-showcase.git
cd imputation-showcase
poetry install
```

## Quick Start

```python
import pandas as pd
import numpy as np
from imputation_showcase.imputation_methods import MeanImputer, KNNImputerMethod

# Create a sample dataset with missing values
df = pd.DataFrame({
    'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
    'feature2': [5.0, np.nan, np.nan, 8.0, 10.0],
    'feature3': [2.0, 4.0, 6.0, np.nan, 10.0]
})

# Apply mean imputation
mean_imputer = MeanImputer()
df_mean = mean_imputer.impute(df)

# Apply KNN imputation
knn_imputer = KNNImputerMethod(n_neighbors=3)
df_knn = knn_imputer.impute(df)

print("Original data:\n", df)
print("\nMean imputed:\n", df_mean)
print("\nKNN imputed:\n", df_knn)
```

## Available Methods

### Statistical Methods

| Method | Class | Description |
|--------|-------|-------------|
| Mean Imputation | `MeanImputer` | Replace missing values with column means |
| Median Imputation | `MedianImputer` | Replace missing values with column medians |
| Mode Imputation | `ModeImputer` | Replace with most frequent value (ideal for categorical/discrete data) |
| Constant Imputation | `ConstantImputer` | Fill with user-specified constant value(s) |
| Quantile Imputation | `QuantileImputer` | Impute using any quantile of observed values |
| Random Sampling | `RandomSamplingImputer` | Randomly sample from observed values to preserve distribution |
| EM Algorithm | `EMImputer` | Expectation-Maximization algorithm assuming multivariate normality |
| End of Distribution | `EndOfDistributionImputer` | Impute at distribution edges (mean ± k*std) for flagging extremes |
| Group Mean | `GroupMeanImputer` | Group-wise mean/median imputation for panel data |

### Time Series Methods

| Method | Class | Description |
|--------|-------|-------------|
| LOCF | `LOCFImputer` | Last Observation Carried Forward |
| NOCB | `NOCBImputer` | Next Observation Carried Backward |
| Interpolation | `InterpolationImputer` | Linear, polynomial, or spline interpolation |
| Moving Average | `MovingAverageImputer` | Fill with rolling window mean/median |
| Weighted Moving Average | `WeightedMovingAverageImputer` | Exponentially weighted moving average (EWMA) for time series |
| Linear Trend | `LinearTrendImputer` | Fit and extrapolate linear trend from observed data |
| Polynomial Trend | `PolynomialTrendImputer` | Fit higher-order polynomial trends for non-linear patterns |
| Seasonal | `SeasonalImputer` | Use seasonal patterns for imputation |
| Kalman Filter | `KalmanFilterImputer` | Kalman filtering with uncertainty estimates for sensor data |
| Forward Fill + Fallback | `ForwardFillFallbackImputer` | LOCF with mean/median fallback for leading NaNs |

### Distance-Based Methods

| Method | Class | Description |
|--------|-------|-------------|
| K-Nearest Neighbors | `KNNImputerMethod` | Impute using weighted average of k-nearest neighbors |
| Hot Deck | `HotDeckImputer` | Random sampling from similar complete observations |

### Regression-Based Methods

| Method | Class | Description |
|--------|-------|-------------|
| Regression | `RegressionImputer` | Linear regression on observed values |
| Stochastic Regression | `StochasticRegressionImputer` | Regression with added noise to preserve variance |
| PMM | `PMMImputer` | Predictive Mean Matching |
| MICE | `MICEImputer` | Multiple Imputation by Chained Equations |

### Tree-Based Methods

| Method | Class | Description |
|--------|-------|-------------|
| MissForest | `MissForestImputer` | Random forest-based iterative imputation |

### Matrix Completion Methods

| Method | Class | Description |
|--------|-------|-------------|
| SoftImpute | `SoftImputeImputer` | Matrix completion via nuclear-norm regularization |
| Bayesian PCA | `BayesianPCAImputer` | Probabilistic PCA for missing data |

### Deep Learning Methods

| Method | Class | Description |
|--------|-------|-------------|
| Autoencoder | `AutoencoderImputer` | Neural network autoencoder-based imputation |
| GAIN | `GAINImputer` | Generative Adversarial Imputation Networks |

### Advanced Statistical Methods

| Method | Class | Description |
|--------|-------|-------------|
| Gaussian Process | `GaussianProcessImputer` | GP regression for missing value prediction |
| Indicator Method | `IndicatorImputer` | Adds binary indicator columns for missingness + imputation |
| Cold Deck | `ColdDeckImputer` | Use predetermined reference values from historical/external data |
| Hybrid | `HybridImputer` | Combines multiple methods with fallback chain for robust imputation |

## Usage Examples

### Basic Imputation

```python
from imputation_showcase.imputation_methods import MedianImputer
import pandas as pd
import numpy as np

# Load your data
df = pd.read_csv("data/example.csv")

# Apply median imputation
imputer = MedianImputer()
df_imputed = imputer.impute(df)
```

### Advanced: Comparing Multiple Methods

```python
from imputation_showcase.imputation_methods import (
    MeanImputer,
    KNNImputerMethod,
    MICEImputer,
    MissForestImputer
)
import pandas as pd

# Load data
df_with_missing = pd.read_csv("data/example.csv")

# Compare different imputation methods
methods = {
    'Mean': MeanImputer(),
    'KNN': KNNImputerMethod(n_neighbors=5),
    'MICE': MICEImputer(max_iter=10),
    'MissForest': MissForestImputer(max_iter=10)
}

results = {}
for name, imputer in methods.items():
    results[name] = imputer.impute(df_with_missing)
    print(f"{name} imputation completed")
```

### Using Evaluation Metrics

```python
from imputation_showcase.imputation_methods import KNNImputerMethod
from imputation_showcase import rmse, mae
import pandas as pd
import numpy as np

# Assume we have ground truth
df_complete = pd.read_csv("data/complete.csv")
df_with_missing = df_complete.copy()

# Introduce missing values randomly
mask = np.random.rand(*df_complete.shape) < 0.2
df_with_missing[mask] = np.nan

# Impute
imputer = KNNImputerMethod()
df_imputed = imputer.impute(df_with_missing)

# Evaluate only on originally missing values
error_rmse = rmse(df_complete[mask], df_imputed[mask])
error_mae = mae(df_complete[mask], df_imputed[mask])

print(f"RMSE: {error_rmse:.4f}")
print(f"MAE: {error_mae:.4f}")
```

### Custom Imputation Pipeline

```python
from imputation_showcase.imputation_methods import BaseImputer
import pandas as pd

class CustomImputer(BaseImputer):
    """Custom imputation logic."""

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        # Your custom imputation logic here
        result = df.fillna(df.median())
        return result

# Use your custom imputer
imputer = CustomImputer()
df_imputed = imputer.impute(df)
```

## Evaluation Metrics

The library provides utility functions to evaluate imputation quality:

- **`rmse(y_true, y_pred)`**: Root Mean Squared Error
- **`mae(y_true, y_pred)`**: Mean Absolute Error

These metrics help quantify the accuracy of imputation when ground truth is available.

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/DiogoRibeiro7/imputation-showcase.git
cd imputation-showcase

# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell
```

### Code Quality

This project maintains high code quality standards:

```bash
# Run linting
poetry run flake8 .

# Run type checking
poetry run mypy imputation_showcase/

# Format check
poetry run black --check .
```

### Pre-commit Hooks

Install pre-commit hooks to ensure code quality before commits:

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

## Testing

### Run Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run coverage run -m pytest
poetry run coverage report

# Run specific test file
poetry run pytest tests/test_imputation_methods.py

# Run with verbose output
poetry run pytest -v
```

### Test Structure

- `tests/test_imputation_methods.py`: Unit tests for individual imputation methods
- `tests/test_benchmarks.py`: Performance benchmarks
- `tests/test_integration.py`: Integration tests

## Continuous Integration

The project uses GitHub Actions for CI/CD:

- **Linting**: flake8 checks on every push
- **Testing**: pytest runs on Python 3.10, 3.11, and 3.12
- **Coverage**: Automatic coverage reporting
- **Type checking**: mypy validation

## Data Requirements

All imputation methods expect:

- **Pandas DataFrame** as input
- **Numeric columns only** (float or int types)
- **NaN values** to indicate missing data

For categorical data, encode features using one-hot encoding or label encoding before imputation.

## Example Data

An example dataset is provided in `data/example.csv` for quick experimentation and testing.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Reporting bugs
- Suggesting enhancements
- Submitting pull requests
- Code of conduct

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Citation

If you use this library in your research or projects, please cite it:

```bibtex
@software{imputation_showcase,
  author = {Ribeiro, Diogo},
  title = {Imputation Showcase: Comprehensive Missing Data Imputation Techniques},
  year = {2024},
  url = {https://github.com/DiogoRibeiro7/imputation-showcase}
}
```

See [CITATION.cff](CITATION.cff) for more citation formats.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with:
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [scikit-learn](https://scikit-learn.org/) - Machine learning tools
- [fancyimpute](https://github.com/iskandr/fancyimpute) - Advanced imputation methods
- [PPCA](https://github.com/allentran/pca-magic) - Probabilistic PCA

## Resources

- **Documentation**: [GitHub README](https://github.com/DiogoRibeiro7/imputation-showcase#readme)
- **Issue Tracker**: [GitHub Issues](https://github.com/DiogoRibeiro7/imputation-showcase/issues)
- **Source Code**: [GitHub Repository](https://github.com/DiogoRibeiro7/imputation-showcase)

---

**Note**: This library is designed for educational and research purposes. Always validate imputation results for your specific use case before using in production.
