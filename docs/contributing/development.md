# Development Setup

Detailed guide for setting up your development environment.

## Prerequisites

### Required Software

- **Python 3.10 or higher** (CI tests 3.10–3.14)
  ```bash
  python --version  # Check your Python version
  ```

- **Poetry** 2.x - Dependency management
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  # or
  pipx install poetry
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
git clone https://github.com/YOUR_USERNAME/imputation-methods.git
cd imputation-methods
```

3. Add upstream remote:
```bash
git remote add upstream https://github.com/DiogoRibeiro7/imputation-methods.git
git remote -v  # Verify remotes
```

### 2. Install Dependencies

```bash
# Install the package plus the test and lint dependency groups
poetry install

# Optional groups: MkDocs for the documentation, Jupyter for the notebooks
poetry install --with docs
poetry install --with notebooks

# Verify installation
poetry run python -c "from imputation_methods import MeanImputer; print('Success!')"
```

Prefer to skip Poetry? The dependency groups are standard `[dependency-groups]` tables, so pip or uv can install the same development environment into an activated virtual environment:

```bash
pip install -e . --group dev      # requires pip >= 25.1
# or
uv pip install -e . --group dev
```

### 3. Configure Pre-commit Hooks

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Test hooks (optional)
poetry run pre-commit run --all-files
```

The hooks run Ruff (lint and format), mypy and basic file hygiene checks (such as trailing whitespace, line endings, YAML/TOML syntax and merge-conflict markers).

### 4. Verify Setup

```bash
# Run tests
poetry run pytest

# Run linter
poetry run ruff check .

# Run type checker
poetry run mypy
```

If all commands succeed, you're ready to develop!

## Development Tools

### Poetry Commands

```bash
# Add a new runtime dependency
poetry add package-name

# Add a development dependency to a group (test, lint, docs or notebooks)
poetry add --group test package-name

# Update dependencies
poetry update

# Show dependency tree
poetry show --tree

# Run a command inside the project environment
poetry run python
```

### Testing

```bash
# Run all tests (includes doctests in src/imputation_methods)
poetry run pytest

# Run with verbose output
poetry run pytest -v

# Run specific test file
poetry run pytest tests/test_statistical.py

# Run tests matching pattern
poetry run pytest -k "mean"

# Run with coverage
poetry run pytest --cov
poetry run pytest --cov --cov-report=html  # Generate HTML report

# Run the wall-clock benchmark tests (skipped by default)
poetry run pytest -m benchmark
```

### Code Quality

```bash
# Lint with Ruff (includes import sorting)
poetry run ruff check .

# Apply safe autofixes
poetry run ruff check --fix .

# Format check with Ruff
poetry run ruff format --check .

# Actually format code
poetry run ruff format .

# Type checking with mypy (strict mode, configured in pyproject.toml)
poetry run mypy

# Run all checks
poetry run pre-commit run --all-files
```

## Continuous Integration

GitHub Actions runs on every push to `main` and on pull requests:

- **Lint and type check:** `ruff check`, `ruff format --check` and `mypy`
- **Tests:** Python 3.10–3.14 on Linux, plus Windows and macOS, with coverage
- **Minimum dependencies:** tests against the oldest versions allowed by `pyproject.toml`
- **Docs:** `mkdocs build --strict`
- **Package:** builds the sdist and wheel, checks metadata and runs the tests against the built wheel

Releases are published to PyPI from a GitHub Release via trusted publishing.

## IDE Setup

### VS Code

Recommended extensions:

- Python (Microsoft)
- Pylance
- Ruff (Astral Software)
- Mypy Type Checker (Microsoft)
- autoDocstring (set the docstring format to Google)
- GitLens

**settings.json:**
```json
{
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true
  },
  "python.testing.pytestEnabled": true
}
```

### PyCharm

1. Configure Poetry as Python interpreter:
   - Settings → Project → Python Interpreter
   - Add → Poetry Environment → Existing

2. Enable code quality tools:
   - Install the Ruff and Mypy plugins, or add `ruff` and `mypy` under Settings → Tools → External Tools

3. Configure pytest:
   - Settings → Tools → Python Integrated Tools
   - Default test runner: pytest
   - Docstring format: Google

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
import numpy as np
import pandas as pd
import pytest

from imputation_methods import MyNewImputer


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
        observed_mask = df['a'].notna()
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
poetry run pytest -m benchmark  # Run only the benchmark tests
```

## Documentation

### Building Docs Locally

```bash
# Install MkDocs (the optional docs group)
poetry install --with docs

# Serve docs locally
poetry run mkdocs serve

# Open browser to http://localhost:8000

# Build docs (CI uses --strict, which fails on warnings such as broken links)
poetry run mkdocs build --strict
```

### Docstring Format

Use Google-style docstrings. Examples in docstrings under `src/imputation_methods` run as doctests with `poetry run pytest`, so keep their output accurate.

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
    return param1 > 0 and bool(param2)
```

## Debugging

### Using pdb

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use built-in breakpoint()
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
# Check which environment Poetry uses
poetry env info

# Reinstall the package in editable mode
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
poetry add --group test pytest-profiling
poetry run pytest --profile

# Profile with cProfile
poetry run python -m cProfile -s cumulative -m pytest
```

### Memory Profiling

```bash
# Install memory-profiler
poetry add --group test memory-profiler

# Profile memory
poetry run python -m memory_profiler script.py
```

## Next Steps

- Review [Code Style Guidelines](code-style.md)
- Check [Contributing Guidelines](guidelines.md)
- Start with issues labeled ["good first issue"](https://github.com/DiogoRibeiro7/imputation-methods/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

Happy coding! 🚀
