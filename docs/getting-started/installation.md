# Installation

## Requirements

- Python 3.10 or higher
- pip or Poetry package manager

## Installation Methods

### Using pip (Recommended for Users)

The simplest way to install Imputation Showcase:

```bash
pip install imputation-showcase
```

### Using Poetry (Recommended for Developers)

If you're developing or want better dependency management:

```bash
poetry add imputation-showcase
```

### From Source

For the latest development version:

```bash
# Clone the repository
git clone https://github.com/DiogoRibeiro7/imputation-showcase.git
cd imputation-showcase

# Install with Poetry
poetry install

# Or install with pip
pip install -e .
```

## Dependencies

The library automatically installs these core dependencies:

- **pandas** ^2.0 - Data manipulation
- **numpy** ^1.24 - Numerical computing
- **scikit-learn** ^1.4 - Machine learning tools
- **fancyimpute** 0.7.0 - Advanced imputation methods
- **ppca** ^0.0.4 - Probabilistic PCA
- **matplotlib** ^3.7 - Visualization
- **seaborn** ^0.12 - Statistical visualization

## Verification

Verify your installation:

```python
import imputation_showcase
from imputation_showcase.imputation_methods import MeanImputer

print("✓ Imputation Showcase installed successfully!")
```

## Development Installation

For contributors who want to run tests and modify the code:

```bash
# Clone and install with dev dependencies
git clone https://github.com/DiogoRibeiro7/imputation-showcase.git
cd imputation-showcase
poetry install

# Install pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest

# Check code quality
poetry run flake8 imputation_showcase tests
```

## Troubleshooting

### Issue: Installation fails with dependency conflicts

**Solution**: Create a fresh virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install imputation-showcase
```

### Issue: fancyimpute installation fails

**Solution**: Install system dependencies first (Ubuntu/Debian):

```bash
sudo apt-get install libopenblas-dev liblapack-dev
pip install imputation-showcase
```

### Issue: Import errors after installation

**Solution**: Ensure you're using the correct Python environment:

```bash
which python  # Should point to your virtual environment
pip list | grep imputation  # Should show imputation-showcase
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Get started with your first imputation
- [Basic Concepts](concepts.md) - Learn about missing data patterns
- [Methods Overview](../user-guide/methods.md) - Explore all available methods
