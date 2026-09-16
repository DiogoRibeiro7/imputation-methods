# Imputation Methods

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/imputation-methods.svg)](https://pypi.org/project/imputation-methods/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**A unified pandas API for 42 missing-data imputation methods: statistical, time-series, regression, ensemble and matrix-completion.**

</div>

---

## Overview

Missing data is a common challenge in data science and machine learning. `imputation-methods` provides a unified interface for comparing and evaluating different imputation strategies, from simple statistical methods to advanced machine learning approaches.

## Features

✨ **42 Imputation Methods** - From mean/median to MICE, MissForest and SoftImpute

🎯 **Unified API** - Every imputer exposes `impute(df)`, and each has a functional shortcut such as `mean_impute(df)`

🔒 **Type-Safe** - Full type hints (ships `py.typed`), checked with mypy in strict mode

📊 **Evaluation Metrics** - RMSE, MAE for quality assessment

🧪 **Well-Tested** - Test suite and doctests run in CI on Python 3.10–3.14

📚 **Extensive Documentation** - Tutorials, examples, and API docs

## Quick Example

```python
import pandas as pd
import numpy as np
from imputation_methods import KNNImputerMethod

# Create data with missing values
df = pd.DataFrame({
    'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
    'feature2': [5.0, np.nan, np.nan, 8.0, 10.0],
})

# Apply KNN imputation
imputer = KNNImputerMethod(k=3)
df_imputed = imputer.impute(df)

print(df_imputed)
```

## Available Methods

### Statistical Methods
- Mean, Median, Mode Imputation
- Constant, Quantile and Trimmed Mean Imputation
- End-of-Distribution Imputation
- Group Mean/Median Imputation
- Missing Indicator + Imputation

### Sampling Methods
- Random Sampling
- Hot Deck
- Cold Deck

### Time Series Methods
- LOCF (Last Observation Carried Forward)
- NOCB (Next Observation Carried Backward)
- Forward Fill with Mean/Median Fallback
- Interpolation (linear, polynomial, spline)
- Moving Average and Exponentially Weighted Moving Average
- Linear and Polynomial Trend
- Seasonal Imputation
- Kalman Filter

### Distance-Based Methods
- K-Nearest Neighbors
- Radius Neighbors
- Local Weighted Mean

### Regression-Based Methods
- Linear Regression
- Stochastic Regression
- Predictive Mean Matching (PMM)
- Bayesian Ridge
- Huber and RANSAC Robust Regression
- Gaussian Process

### Iterative Methods
- MICE (Multiple Imputation by Chained Equations)
- MissForest
- EM-style iterative imputation (chained equations, not closed-form multivariate-normal EM)

### Matrix Completion Methods
- SoftImpute
- Bayesian PCA (maximum-likelihood probabilistic PCA)

### Neural Network Methods
- Autoencoder
- GAIN (Generative Adversarial Imputation Nets)

### Ensemble Methods
- Hybrid (fallback chain)
- Stacking (mean/median of several imputers)
- Bagging (bootstrap aggregating of a base imputer)

## Installation

=== "Using pip"

    ```bash
    pip install imputation-methods

    # With the optional plotting dependencies (matplotlib, seaborn)
    pip install "imputation-methods[viz]"
    ```

=== "Using Poetry"

    ```bash
    poetry add imputation-methods
    ```

=== "From Source"

    ```bash
    git clone https://github.com/DiogoRibeiro7/imputation-methods.git
    cd imputation-methods
    poetry install
    ```

## Next Steps

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Getting Started__

    ---

    Learn the basics with our comprehensive tutorial

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } __User Guide__

    ---

    Explore the imputation methods and when to use them

    [:octicons-arrow-right-24: Methods Overview](user-guide/methods.md)

-   :material-code-braces:{ .lg .middle } __Examples__

    ---

    See practical examples for real-world scenarios

    [:octicons-arrow-right-24: View Examples](examples/time-series.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Detailed API documentation for all methods

    [:octicons-arrow-right-24: API Docs](api/index.md)

</div>

## Community

- **GitHub**: [Issues](https://github.com/DiogoRibeiro7/imputation-methods/issues) and [Discussions](https://github.com/DiogoRibeiro7/imputation-methods/discussions)
- **Email**: [diogo.debastos.ribeiro@gmail.com](mailto:diogo.debastos.ribeiro@gmail.com)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/DiogoRibeiro7/imputation-methods/blob/main/LICENSE) file for details.

## Citation

If you use this library in your research:

```bibtex
@software{imputation_methods,
  author = {Ribeiro, Diogo},
  title = {imputation-methods: A Unified pandas API for Missing-Data Imputation},
  year = {2024},
  url = {https://github.com/DiogoRibeiro7/imputation-methods}
}
```
