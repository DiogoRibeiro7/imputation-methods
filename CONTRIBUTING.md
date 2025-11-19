# Contributing to Imputation Showcase

First off, thank you for considering contributing to Imputation Showcase! It's people like you that make this project better for everyone. We welcome contributions from the community and are pleased to have you join us.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Adding New Imputation Methods](#adding-new-imputation-methods)
  - [Improving Documentation](#improving-documentation)
  - [Code Contributions](#code-contributions)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [Getting Help](#getting-help)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [diogo.debastos.ribeiro@gmail.com](mailto:diogo.debastos.ribeiro@gmail.com).

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the [issue tracker](https://github.com/DiogoRibeiro7/imputation-showcase/issues) to avoid duplicates. When you create a bug report, include as many details as possible:

**Template for Bug Reports:**

```markdown
**Description:**
A clear and concise description of the bug.

**To Reproduce:**
Steps to reproduce the behavior:
1. Import module '...'
2. Call function '....'
3. See error

**Expected Behavior:**
What you expected to happen.

**Actual Behavior:**
What actually happened.

**Environment:**
- OS: [e.g., Ubuntu 22.04, macOS 13.0, Windows 11]
- Python version: [e.g., 3.10.5]
- Package version: [e.g., 0.1.0]
- Installation method: [e.g., pip, poetry, from source]

**Additional Context:**
Any other information about the problem.

**Stack Trace:**
```python
# Paste full error traceback here
```
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Clear title and description** of the suggested enhancement
- **Use case** - explain why this would be useful
- **Possible implementation** - if you have ideas on how to implement it
- **Alternatives considered** - what other solutions you've thought about

### Adding New Imputation Methods

We're always interested in adding new imputation techniques! If you'd like to contribute a new method:

1. **Check existing methods** - ensure it's not already implemented
2. **Research the method** - understand the algorithm thoroughly
3. **Follow the pattern** - inherit from `BaseImputer` and implement `impute()`
4. **Add tests** - comprehensive unit tests are required
5. **Document it** - add docstrings with examples
6. **Update README** - add the method to the Available Methods table

**Example Structure:**

```python
class NewImputer(BaseImputer):
    """Brief description of the imputation method.

    Longer description explaining:
    - How the method works
    - When to use it
    - Advantages and disadvantages

    Parameters:
        param1: Description of parameter 1
        param2: Description of parameter 2

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import NewImputer
        >>> df = pd.DataFrame({"a": [1, 2, np.nan, 4]})
        >>> imputer = NewImputer()
        >>> imputed = imputer.impute(df)
    """

    def __init__(self, param1=default1, param2=default2):
        self.param1 = param1
        self.param2 = param2

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        # Implementation here
        return result
```

### Improving Documentation

Documentation improvements are always welcome! This includes:

- Fixing typos or clarifying existing documentation
- Adding examples to docstrings
- Improving the README
- Creating tutorials or guides
- Adding code comments for complex logic

### Code Contributions

We actively welcome your pull requests for:

- Bug fixes
- New features
- Performance improvements
- Code refactoring
- Test improvements

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [Poetry](https://python-poetry.org/docs/) for dependency management
- Git for version control

### Setting Up Your Environment

1. **Fork the repository** on GitHub

2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/imputation-showcase.git
   cd imputation-showcase
   ```

3. **Add upstream remote:**
   ```bash
   git remote add upstream https://github.com/DiogoRibeiro7/imputation-showcase.git
   ```

4. **Install dependencies:**
   ```bash
   poetry install
   ```

5. **Activate the virtual environment:**
   ```bash
   poetry shell
   ```

6. **Install pre-commit hooks:**
   ```bash
   poetry run pre-commit install
   ```

7. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

## Coding Standards

### Python Style Guide

This project follows [PEP 8](https://pep8.org/) with some modifications:

- **Line length:** Maximum 88 characters (Black default)
- **Imports:** Organized with `isort`
- **Docstrings:** Google-style format
- **Type hints:** Required for all function signatures

### Code Quality Tools

We use several tools to maintain code quality:

```bash
# Run linting
poetry run flake8 imputation_showcase tests

# Run type checking
poetry run mypy imputation_showcase/

# Run all pre-commit hooks
poetry run pre-commit run --all-files
```

### Type Hints

All functions should include type hints:

```python
def impute(self, df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values."""
    # implementation
```

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """Brief description of what the function does.

    Longer description if needed, explaining the behavior,
    algorithm, or any important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When and why this is raised
        TypeError: When and why this is raised

    Examples:
        >>> example_function(42, "test")
        True
    """
```

## Testing Guidelines

### Writing Tests

- **Location:** Place tests in `tests/` directory
- **Naming:** Test files should be named `test_*.py`
- **Coverage:** Aim for >80% code coverage
- **Test structure:** Use pytest fixtures and parametrize when appropriate

### Test Categories

1. **Unit tests** - Test individual methods in isolation
2. **Integration tests** - Test multiple components together
3. **Edge cases** - Test boundary conditions and error handling

### Example Test

```python
import pytest
import pandas as pd
import numpy as np
from imputation_showcase import MeanImputer

def test_mean_imputer_basic():
    """Test basic mean imputation functionality."""
    df = pd.DataFrame({
        'a': [1.0, 2.0, np.nan, 4.0],
        'b': [5.0, np.nan, 7.0, 8.0]
    })
    imputer = MeanImputer()
    result = imputer.impute(df)

    assert not result.isna().any().any()
    assert result.loc[2, 'a'] == pytest.approx(2.333, rel=1e-2)

def test_mean_imputer_no_missing():
    """Test mean imputer with no missing values."""
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
    imputer = MeanImputer()
    result = imputer.impute(df)

    pd.testing.assert_frame_equal(result, df)
```

### Running Tests

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

# Run tests matching a pattern
poetry run pytest -k "mean"
```

## Documentation Standards

### Code Documentation

- Every public class and function must have a docstring
- Docstrings should include examples when appropriate
- Complex algorithms should have inline comments explaining the logic

### README Updates

If your change affects the public API or adds new features:

1. Update the relevant sections in README.md
2. Add usage examples
3. Update the Available Methods table if adding new imputers

### CHANGELOG Updates

For significant changes, add an entry to CHANGELOG.md under "Unreleased" section:

```markdown
### Added
- New imputation method: XYZ Imputer

### Changed
- Improved performance of KNN imputation by 20%

### Fixed
- Bug in MICE imputer with single column dataframes
```

## Pull Request Process

### Before Submitting

1. **Update your branch** with latest upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks:**
   ```bash
   poetry run flake8 .
   poetry run mypy imputation_showcase/
   poetry run pytest
   poetry run coverage run -m pytest
   ```

3. **Update documentation** if needed

4. **Add tests** for new functionality

### Submitting Your PR

1. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub with a clear title and description

3. **Fill out the PR template** completely

4. **Link related issues** using keywords (e.g., "Fixes #123")

### PR Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Coverage maintained or improved

## Checklist
- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added and passing
- [ ] CHANGELOG.md updated (if applicable)

## Additional Notes
Any additional information, context, or screenshots.
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by maintainers
3. **Requested changes** should be addressed
4. **Approval** by at least one maintainer
5. **Merge** by maintainers

## Getting Help

### Resources

- **Documentation:** [README.md](README.md)
- **Issues:** [GitHub Issues](https://github.com/DiogoRibeiro7/imputation-showcase/issues)
- **Discussions:** [GitHub Discussions](https://github.com/DiogoRibeiro7/imputation-showcase/discussions)

### Contact

- **Email:** [diogo.debastos.ribeiro@gmail.com](mailto:diogo.debastos.ribeiro@gmail.com)
- **Issues:** For bug reports and feature requests

### Questions?

Don't hesitate to ask questions by:
- Opening a [GitHub Discussion](https://github.com/DiogoRibeiro7/imputation-showcase/discussions)
- Creating an issue labeled "question"
- Emailing the maintainer

## Recognition

Contributors will be recognized in:
- The project's README (if significant contribution)
- Release notes for the version including their contribution
- GitHub's automatic contributor tracking

Thank you for contributing to Imputation Showcase! Your efforts help make this project better for everyone.

---

**Note:** This is a living document. If you have suggestions for improving these guidelines, please open an issue or submit a pull request.
