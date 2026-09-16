# Contributing Guidelines

Thank you for considering contributing to `imputation-methods`! This page provides guidelines for contributing to the project.

## Quick Links

- **Main Contributing Guide:** [CONTRIBUTING.md](https://github.com/DiogoRibeiro7/imputation-methods/blob/main/CONTRIBUTING.md)
- **Code of Conduct:** [CODE_OF_CONDUCT.md](https://github.com/DiogoRibeiro7/imputation-methods/blob/main/CODE_OF_CONDUCT.md)
- **Issues:** [GitHub Issues](https://github.com/DiogoRibeiro7/imputation-methods/issues)

## How to Contribute

### 1. Reporting Bugs

Found a bug? Please open an issue with:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, package version)
- Full error traceback if applicable

### 2. Suggesting Features

Have an idea? We'd love to hear it! Open an issue describing:

- The problem your feature would solve
- Proposed solution or API
- Alternative approaches you considered
- Use cases and examples

### 3. Adding Imputation Methods

Want to add a new imputation technique? Great! Please:

1. Inherit from `BaseImputer` and put the class in the matching module under `src/imputation_methods/` (for example `statistical.py`, `time_series.py` or `regression.py`)
2. Implement the `impute()` method
3. Add a functional shortcut in `functional.py` and export both from `src/imputation_methods/__init__.py` (including `__all__`)
4. Add comprehensive tests
5. Document with Google-style docstrings and examples (docstring examples run as doctests)
6. Update README and documentation

**Example template:**

```python
import pandas as pd

# Inside the package, use a relative import: from .base import BaseImputer
from imputation_methods import BaseImputer


class MyNewImputer(BaseImputer):
    """Brief description.

    Detailed explanation of:
    - How it works
    - When to use it
    - Pros and cons

    Args:
        param1: Description

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4]})
        >>> imputer = MyNewImputer()
        >>> result = imputer.impute(df)
    """

    def __init__(self, param1: float = 1.0) -> None:
        self.param1 = param1

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        result = df.copy()
        # Your implementation
        return result
```

### 4. Improving Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples
- Improve docstrings
- Create tutorials

### 5. Code Contributions

We welcome pull requests for:

- Bug fixes
- Performance improvements
- New features
- Code refactoring
- Test improvements

## Getting Started

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependency management (or pip >= 25.1 / uv, see [Development Setup](development.md))
- Git

### Setup

1. Fork the repository on GitHub

2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/imputation-methods.git
cd imputation-methods
```

3. Add upstream remote:
```bash
git remote add upstream https://github.com/DiogoRibeiro7/imputation-methods.git
```

4. Install the package plus the `test` and `lint` dependency groups:
```bash
poetry install
```

5. Install pre-commit hooks:
```bash
poetry run pre-commit install
```

6. Create a branch:
```bash
git checkout -b feature/your-feature-name
```

## Development Workflow

### 1. Make Your Changes

Edit code, tests, and documentation as needed.

### 2. Run Code Quality Checks

```bash
# Linting
poetry run ruff check .

# Formatting
poetry run ruff format .

# Type checking (strict mode, configured in pyproject.toml)
poetry run mypy

# All pre-commit hooks (Ruff lint + format, mypy, file hygiene)
poetry run pre-commit run --all-files
```

### 3. Run Tests

```bash
# All tests (includes doctests in src; benchmark tests are skipped)
poetry run pytest

# With coverage
poetry run pytest --cov

# Wall-clock benchmark tests
poetry run pytest -m benchmark

# Specific test
poetry run pytest tests/test_imputation_methods.py -v
```

### 4. Update Documentation

If you changed the API or added features:

- Update docstrings
- Update README.md
- Update CHANGELOG.md

### 5. Commit Your Changes

```bash
git add .
git commit -m "Clear description of changes"
```

Use meaningful commit messages following this format:

- `feat: Add new XYZ imputation method`
- `fix: Resolve issue with KNN imputer on single column`
- `docs: Improve README examples`
- `test: Add tests for edge cases in MICE`
- `refactor: Simplify BaseImputer logic`

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

- Clear title and description
- Link to related issues
- Description of testing done
- Screenshots if applicable

## Pull Request Checklist

Before submitting your PR, ensure:

- [ ] Code follows project style (passes `ruff check`, `ruff format --check` and `mypy`)
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for significant changes)
- [ ] Branch is up to date with upstream/main
- [ ] Commit messages are clear and descriptive

## Code Review Process

1. Automated checks run in GitHub Actions: Ruff and mypy, tests on Python 3.10–3.14 on Linux plus Windows and macOS, a minimum-dependency-versions job, the docs build and package build checks
2. Maintainers review code
3. Discussion and requested changes
4. Approval by maintainer
5. Merge to main branch

Please be patient - reviews may take a few days. We'll do our best to provide constructive feedback.

## Recognition

All contributors are recognized in:

- Project README (for significant contributions)
- Release notes
- GitHub contributor tracking

## Getting Help

Need help? You can:

- Open a [GitHub Discussion](https://github.com/DiogoRibeiro7/imputation-methods/discussions)
- Create an issue labeled "question"
- Email: diogo.debastos.ribeiro@gmail.com

## Next Steps

- Read [Development Setup](development.md) for detailed environment setup
- Review [Code Style](code-style.md) for coding standards
- Check out [existing issues](https://github.com/DiogoRibeiro7/imputation-methods/issues) labeled "good first issue"

Thank you for contributing! 🎉
