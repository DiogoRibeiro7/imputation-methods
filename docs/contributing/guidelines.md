# Contributing Guidelines

Thank you for considering contributing to Imputation Showcase! This page provides guidelines for contributing to the project.

## Quick Links

- **Main Contributing Guide:** [CONTRIBUTING.md](https://github.com/DiogoRibeiro7/imputation-showcase/blob/main/CONTRIBUTING.md)
- **Code of Conduct:** [CODE_OF_CONDUCT.md](https://github.com/DiogoRibeiro7/imputation-showcase/blob/main/CODE_OF_CONDUCT.md)
- **Issues:** [GitHub Issues](https://github.com/DiogoRibeiro7/imputation-showcase/issues)

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

1. Inherit from `BaseImputer`
2. Implement the `impute()` method
3. Add comprehensive tests
4. Document with examples
5. Update README and documentation

**Example template:**

```python
from imputation_showcase import BaseImputer
import pandas as pd

class MyNewImputer(BaseImputer):
    """Brief description.

    Detailed explanation of:
    - How it works
    - When to use it
    - Pros and cons

    Args:
        param1: Description
        param2: Description

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4]})
        >>> imputer = MyNewImputer()
        >>> imputer.impute(df)
    """

    def __init__(self, param1=default):
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
- [Poetry](https://python-poetry.org/) for dependency management
- Git

### Setup

1. Fork the repository on GitHub

2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/imputation-showcase.git
cd imputation-showcase
```

3. Add upstream remote:
```bash
git remote add upstream https://github.com/DiogoRibeiro7/imputation-showcase.git
```

4. Install dependencies:
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
poetry run flake8 imputation_showcase tests

# Type checking
poetry run mypy imputation_showcase/

# Formatting
poetry run black --check .

# All pre-commit hooks
poetry run pre-commit run --all-files
```

### 3. Run Tests

```bash
# All tests
poetry run pytest

# With coverage
poetry run coverage run -m pytest
poetry run coverage report

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

- [ ] Code follows project style (passes flake8, mypy, black)
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for significant changes)
- [ ] Branch is up to date with upstream/main
- [ ] Commit messages are clear and descriptive

## Code Review Process

1. Automated checks run (CI/CD)
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

- Open a [GitHub Discussion](https://github.com/DiogoRibeiro7/imputation-showcase/discussions)
- Create an issue labeled "question"
- Email: diogo.debastos.ribeiro@gmail.com

## Next Steps

- Read [Development Setup](development.md) for detailed environment setup
- Review [Code Style](code-style.md) for coding standards
- Check out [existing issues](https://github.com/DiogoRibeiro7/imputation-showcase/issues) labeled "good first issue"

Thank you for contributing! 🎉
