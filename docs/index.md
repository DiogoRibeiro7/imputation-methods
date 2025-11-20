# Imputation Showcase

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-black.svg)](https://flake8.pycqa.org/)

**A comprehensive Python library showcasing 16+ state-of-the-art imputation techniques for handling missing data.**

</div>

---

## Overview

Missing data is a common challenge in data science and machine learning. Imputation Showcase provides a unified interface for comparing and evaluating different imputation strategies, from simple statistical methods to advanced machine learning approaches.

## Features

✨ **16+ Imputation Methods** - From mean/median to MICE and GAIN

🎯 **Unified API** - Consistent interface across all methods

🔒 **Type-Safe** - Full type hints for better IDE support

📊 **Evaluation Metrics** - RMSE, MAE for quality assessment

🧪 **Well-Tested** - Comprehensive test suite with >80% coverage

📚 **Extensive Documentation** - Tutorials, examples, and API docs

## Quick Example

```python
import pandas as pd
import numpy as np
from imputation_showcase.imputation_methods import KNNImputerMethod

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
- Mean Imputation
- Median Imputation

### Time Series Methods
- LOCF (Last Observation Carried Forward)
- NOCB (Next Observation Carried Backward)

### Distance-Based Methods
- K-Nearest Neighbors
- Hot Deck

### Regression-Based Methods
- Linear Regression
- Stochastic Regression
- Predictive Mean Matching (PMM)
- MICE (Multiple Imputation by Chained Equations)

### Tree-Based Methods
- MissForest

### Matrix Completion Methods
- SoftImpute
- Bayesian PCA

### Deep Learning Methods
- Autoencoder
- GAIN (Generative Adversarial Imputation Networks)

### Advanced Statistical Methods
- Gaussian Process

## Installation

=== "Using pip"

    ```bash
    pip install imputation-showcase
    ```

=== "Using Poetry"

    ```bash
    poetry add imputation-showcase
    ```

=== "From Source"

    ```bash
    git clone https://github.com/DiogoRibeiro7/imputation-showcase.git
    cd imputation-showcase
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

    Explore all imputation methods and when to use them

    [:octicons-arrow-right-24: Methods Overview](user-guide/methods.md)

-   :material-code-braces:{ .lg .middle } __Examples__

    ---

    See practical examples for real-world scenarios

    [:octicons-arrow-right-24: View Examples](examples/time-series.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Detailed API documentation for all methods

    [:octicons-arrow-right-24: API Docs](api/methods.md)

</div>

## Community

- **GitHub**: [Issues](https://github.com/DiogoRibeiro7/imputation-showcase/issues) and [Discussions](https://github.com/DiogoRibeiro7/imputation-showcase/discussions)
- **Email**: [diogo.debastos.ribeiro@gmail.com](mailto:diogo.debastos.ribeiro@gmail.com)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/DiogoRibeiro7/imputation-showcase/blob/main/LICENSE) file for details.

## Citation

If you use this library in your research:

```bibtex
@software{imputation_showcase,
  author = {Ribeiro, Diogo},
  title = {Imputation Showcase: Comprehensive Missing Data Imputation Techniques},
  year = {2024},
  url = {https://github.com/DiogoRibeiro7/imputation-showcase}
}
```
