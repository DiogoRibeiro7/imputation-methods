# Installation

## Requirements

- Python 3.10 or higher (3.10–3.14 are supported)
- pip or Poetry package manager

## Installation Methods

### Using pip (Recommended for Users)

The simplest way to install `imputation-methods`:

```bash
pip install imputation-methods
```

The notebooks, example scripts and plotting snippets in these docs also use matplotlib and seaborn. Install them with the `viz` extra:

```bash
pip install "imputation-methods[viz]"
```

### Using Poetry (Recommended for Developers)

If you're developing or want better dependency management:

```bash
poetry add imputation-methods

# With the plotting extra
poetry add "imputation-methods[viz]"
```

### From Source

For the latest development version:

```bash
# Clone the repository
git clone https://github.com/DiogoRibeiro7/imputation-methods.git
cd imputation-methods

# Install with Poetry
poetry install

# Or install with pip
pip install -e .
```

## Dependencies

The library automatically installs these core dependencies:

- **numpy** >=1.24 - Numerical computing
- **pandas** >=2.0 - Data manipulation (pandas 3 is supported)
- **scipy** >=1.10 - Scientific computing
- **scikit-learn** >=1.4 - Machine learning tools

Optional `viz` extra:

- **matplotlib** >=3.7 - Visualization
- **seaborn** >=0.13 - Statistical visualization

SoftImpute and Bayesian PCA are implemented directly on NumPy, so no extra matrix-completion packages are required.

## Verification

Verify your installation:

```python
import imputation_methods
from imputation_methods import MeanImputer

print(f"✓ imputation-methods {imputation_methods.__version__} installed successfully!")
```

## Development Installation

For contributors who want to run tests and modify the code:

```bash
# Clone and install the package plus the test and lint dependency groups
git clone https://github.com/DiogoRibeiro7/imputation-methods.git
cd imputation-methods
poetry install

# Install pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest

# Check code quality
poetry run ruff check .
poetry run mypy
```

Without Poetry, install the same development dependencies with `pip install -e . --group dev` (requires pip >= 25.1) or `uv pip install -e . --group dev`. See [Development Setup](../contributing/development.md) for details.

## Troubleshooting

### Issue: Installation fails with dependency conflicts

**Solution**: Create a fresh virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install imputation-methods
```

### Issue: `ModuleNotFoundError` for matplotlib or seaborn

**Solution**: The plotting libraries are optional. Install the `viz` extra:

```bash
pip install "imputation-methods[viz]"
```

### Issue: Import errors after installation

**Solution**: Ensure you're using the correct Python environment. Note that the distribution is named `imputation-methods`, but the import name is `imputation_methods`:

```bash
which python  # Should point to your virtual environment
pip list | grep imputation  # Should show imputation-methods
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Get started with your first imputation
- [Basic Concepts](concepts.md) - Learn about missing data patterns
- [Methods Overview](../user-guide/methods.md) - Explore all available methods
