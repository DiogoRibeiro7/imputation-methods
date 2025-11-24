# Development Setup

Detailed guide for setting up your development environment.

## Prerequisites

### Required Software

- **Python 3.10 or higher**
  ```bash
  python --version  # Check your Python version
  ```

- **Poetry** - Dependency management
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  # or
  pip install --user poetry
  ```

- **Git** - Version control
  ```bash
  git --version
  ```

## Initial Setup

### 1. Fork and Clone

1. Fork the repository on GitHub (click "Fork" button)

2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/imputation-showcase.git
cd imputation-showcase
```

3. Add upstream remote:
```bash
git remote add upstream https://github.com/DiogoRibeiro7/imputation-showcase.git
git remote -v  # Verify remotes
```

### 2. Install Dependencies

```bash
# Install all dependencies (including dev dependencies)
poetry install

# Activate virtual environment
poetry shell

# Verify installation
poetry run python -c "from imputation_showcase import MeanImputer; print('Success!')"
```

### 3. Configure Pre-commit Hooks

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Test hooks (optional)
poetry run pre-commit run --all-files
```

### 4. Verify Setup

```bash
# Run tests
poetry run pytest

# Run linter
poetry run flake8 imputation_showcase tests

# Run type checker
poetry run mypy imputation_showcase/
```

If all commands succeed, you're ready to develop!

## Development Tools

### Poetry Commands

```bash
# Add a new dependency
poetry add package-name

# Add a dev dependency
poetry add --group dev package-name

# Update dependencies
poetry update

# Show dependency tree
poetry show --tree

# Export requirements.txt
poetry export -f requirements.txt --output requirements.txt
```

### Testing

```bash
# Run all tests
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run specific test file
poetry run pytest tests/test_imputation_methods.py

# Run tests matching pattern
poetry run pytest -k "mean"

# Run with coverage
poetry run coverage run -m pytest
poetry run coverage report
poetry run coverage html  # Generate HTML report

# Run tests in parallel (faster)
poetry run pytest -n auto
```

### Code Quality

```bash
# Linting with flake8
poetry run flake8 imputation_showcase tests

# Type checking with mypy
poetry run mypy imputation_showcase/

# Format check with black
poetry run black --check .

# Actually format code
poetry run black .

# Import sorting
poetry run isort --check-only .
poetry run isort .  # Actually sort

# Run all checks
poetry run pre-commit run --all-files
```

## IDE Setup

### VS Code

Recommended extensions:

- Python (Microsoft)
- Pylance
- Black Formatter
- autoDocstring
- GitLens

**settings.json:**
```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackPath": "poetry run black",
  "editor.formatOnSave": true,
  "python.linting.mypyEnabled": true,
  "python.testing.pytestEnabled": true
}
```

### PyCharm

1. Configure Poetry as Python interpreter:
   - Settings → Project → Python Interpreter
   - Add → Poetry Environment → Existing

2. Enable code quality tools:
   - Settings → Tools → External Tools
   - Add flake8, mypy, black

3. Configure pytest:
   - Settings → Tools → Python Integrated Tools
   - Default test runner: pytest

## Working with Git

### Branch Naming Convention

```bash
# Feature branches
git checkout -b feature/add-xyz-imputer
git checkout -b feature/improve-knn-performance

# Bug fix branches
git checkout -b fix/mice-single-column-bug
git checkout -b fix/memory-leak-in-missforest

# Documentation branches
git checkout -b docs/update-readme
git checkout -b docs/add-examples
```

### Keeping Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Update your main branch
git checkout main
git merge upstream/main
git push origin main

# Rebase your feature branch
git checkout feature/your-feature
git rebase main
```

### Commit Message Guidelines

Follow conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting, no code change
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```
feat(imputation): Add trimmed mean imputer

Implement a robust imputation method using trimmed mean
to reduce the influence of outliers.

Closes #123
```

```
fix(knn): Handle single column DataFrames correctly

KNN imputer was failing when DataFrame had only one column.
Added check and fallback to mean imputation for this case.

Fixes #456
```

## Testing Workflow

### Writing Tests

```python
# tests/test_my_feature.py
import pytest
import pandas as pd
import numpy as np
from imputation_showcase import MyNewImputer

class TestMyNewImputer:
    """Tests for MyNewImputer."""

    def test_basic_functionality(self):
        """Test basic imputation."""
        df = pd.DataFrame({'a': [1, 2, np.nan, 4]})
        imputer = MyNewImputer()
        result = imputer.impute(df)
        assert not result.isna().any().any()

    def test_preserves_observed_values(self):
        """Test that observed values are unchanged."""
        df = pd.DataFrame({'a': [1, 2, np.nan, 4]})
        imputer = MyNewImputer()
        result = imputer.impute(df)
        observed_mask = ~df.isna()
        pd.testing.assert_series_equal(
            df.loc[observed_mask, 'a'],
            result.loc[observed_mask, 'a']
        )

    @pytest.mark.parametrize("missing_rate", [0.1, 0.3, 0.5])
    def test_various_missing_rates(self, missing_rate):
        """Test with different missingness levels."""
        # Test implementation
        pass
```

### Running Specific Tests

```bash
# Run single test
poetry run pytest tests/test_my_feature.py::TestMyNewImputer::test_basic_functionality

# Run test class
poetry run pytest tests/test_my_feature.py::TestMyNewImputer

# Run with markers
poetry run pytest -m slow  # Run only slow tests
poetry run pytest -m "not slow"  # Skip slow tests
```

## Documentation

### Building Docs Locally

```bash
# Install MkDocs (should be installed with dev dependencies)
poetry install

# Serve docs locally
poetry run mkdocs serve

# Open browser to http://localhost:8000

# Build docs
poetry run mkdocs build
```

### Docstring Format

```python
def example_function(param1: int, param2: str) -> bool:
    """Brief one-line description.

    More detailed description if needed. Explain the behavior,
    any important details, or algorithm overview.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When and why
        TypeError: When and why

    Examples:
        >>> example_function(42, "test")
        True

        >>> example_function(0, "")
        False
    """
    # Implementation
    pass
```

## Debugging

### Using pdb

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use built-in breakpoint() (Python 3.7+)
breakpoint()
```

### Using pytest with pdb

```bash
# Drop into pdb on failure
poetry run pytest --pdb

# Drop into pdb on first failure, then end
poetry run pytest -x --pdb
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

def my_function():
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
```

## Troubleshooting

### Poetry Issues

```bash
# Clear cache
poetry cache clear pypi --all

# Reinstall dependencies
rm poetry.lock
poetry install

# Update Poetry itself
poetry self update
```

### Import Errors

```bash
# Ensure you're in the virtual environment
poetry shell

# Reinstall package in editable mode
poetry install
```

### Test Failures

```bash
# Run with verbose output
poetry run pytest -vv

# Show print statements
poetry run pytest -s

# Run without capturing output
poetry run pytest --capture=no
```

## Performance Profiling

### Profiling Tests

```bash
# Profile with pytest-profiling
poetry add --group dev pytest-profiling
poetry run pytest --profile

# Profile with cProfile
poetry run python -m cProfile -s cumulative -m pytest
```

### Memory Profiling

```bash
# Install memory-profiler
poetry add --group dev memory-profiler

# Profile memory
poetry run python -m memory_profiler script.py
```

## Next Steps

- Review [Code Style Guidelines](code-style.md)
- Check [Contributing Guidelines](guidelines.md)
- Start with issues labeled ["good first issue"](https://github.com/DiogoRibeiro7/imputation-showcase/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

Happy coding! 🚀
