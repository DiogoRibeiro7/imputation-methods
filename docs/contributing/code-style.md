# Code Style Guide

Comprehensive code style and quality guidelines for the project.

## Overview

This project follows PEP 8 with some modifications for consistency and readability. We use automated tools to enforce style guidelines.

## Style Guidelines

### Line Length

- **Maximum: 88 characters** (Black default)
- Exception: Long URLs or strings that can't be broken

```python
# Good
def short_function_name(param1, param2):
    return param1 + param2

# Good - use parentheses for line continuation
result = some_function(
    argument1,
    argument2,
    argument3
)

# Avoid - line too long
result = some_function(argument1, argument2, argument3, argument4, argument5, argument6)
```

### Imports

Organize imports in this order:
1. Standard library
2. Third-party packages
3. Local application imports

```python
# Standard library
import os
import sys
from typing import Optional, Union

# Third-party
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Local
from imputation_showcase.imputation_methods import BaseImputer
from imputation_showcase import rmse, mae
```

Use `isort` to automatically organize imports:

```bash
poetry run isort imputation_showcase tests
```

### Naming Conventions

#### Classes

- **PascalCase** for class names
- Descriptive and specific

```python
# Good
class KNNImputerMethod:
    pass

class BayesianPCAImputer:
    pass

# Avoid
class knn_imputer:  # Wrong case
    pass

class Imputer:  # Too generic
    pass
```

#### Functions and Methods

- **snake_case** for functions and methods
- Use descriptive verb phrases

```python
# Good
def impute_missing_values(df):
    pass

def calculate_rmse(y_true, y_pred):
    pass

# Avoid
def ImputeMissingValues(df):  # Wrong case
    pass

def calc(y1, y2):  # Not descriptive
    pass
```

#### Variables

- **snake_case** for variables
- Descriptive names

```python
# Good
missing_count = df.isna().sum()
imputed_dataframe = imputer.impute(df)

# Avoid
mc = df.isna().sum()  # Too abbreviated
ImputedDataFrame = imputer.impute(df)  # Wrong case
```

#### Constants

- **UPPER_SNAKE_CASE** for constants

```python
# Good
DEFAULT_K_NEIGHBORS = 5
MAX_ITERATIONS = 100
MIN_OBSERVATIONS = 1

# Module-level constants
_PRIVATE_CONSTANT = 42
```

### Type Hints

All public functions must have type hints:

```python
from typing import Optional, Union
import pandas as pd
import numpy as np

# Good
def impute(self, df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values."""
    pass

def calculate_metric(
    y_true: pd.Series,
    y_pred: pd.Series,
    metric_type: str = "rmse"
) -> float:
    """Calculate evaluation metric."""
    pass

# Complex types
def process_data(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    fill_value: Union[int, float] = 0
) -> tuple[pd.DataFrame, dict]:
    """Process data with options."""
    pass
```

## Docstrings

Use Google-style docstrings for all public classes, methods, and functions.

### Function Docstrings

```python
def example_function(param1: int, param2: str, param3: bool = False) -> dict:
    """Brief one-line description ending with period.

    More detailed description if needed. Explain the purpose,
    behavior, and any important implementation details.

    Args:
        param1: Description of param1. What it represents and
            any constraints or special values.
        param2: Description of param2.
        param3: Description of param3. Defaults to False.

    Returns:
        Description of return value. Explain the structure
        if returning a complex object.

    Raises:
        ValueError: When param1 is negative.
        TypeError: When param2 is not a string.

    Examples:
        >>> result = example_function(42, "test")
        >>> print(result)
        {'key': 'value'}

        >>> example_function(0, "")
        {}

    Note:
        Any additional notes or warnings.
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")

    return {'key': 'value'}
```

### Class Docstrings

```python
class ExampleImputer(BaseImputer):
    """Brief description of the imputation method.

    Detailed explanation of:
    - How the method works
    - When to use it
    - Advantages and disadvantages
    - References to papers (if applicable)

    Args:
        param1: Description of initialization parameter.
        param2: Description of another parameter.

    Attributes:
        attribute1: Description of instance attribute.
        attribute2: Description of another attribute.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4]})
        >>> imputer = ExampleImputer(param1=5)
        >>> result = imputer.impute(df)
        >>> print(result)

    References:
        - Author Name. "Paper Title." Journal, Year.
        - https://example.com/documentation
    """

    def __init__(self, param1: int, param2: str = "default"):
        """Initialize the imputer.

        Args:
            param1: Description.
            param2: Description. Defaults to "default".
        """
        self.param1 = param1
        self.param2 = param2
```

## Code Organization

### File Structure

```python
"""Module docstring describing the file's purpose."""

# Imports (organized by category)
import os
from typing import Optional

import pandas as pd
import numpy as np

from imputation_showcase import BaseImputer

# Constants
DEFAULT_VALUE = 42
_PRIVATE_CONSTANT = 100

# Classes
class MyClass:
    """Class implementation."""
    pass

# Functions
def helper_function():
    """Helper function."""
    pass

# Main execution (if applicable)
if __name__ == "__main__":
    main()
```

### Method Order in Classes

```python
class Example(BaseImputer):
    """Example class."""

    # 1. Class variables
    class_variable = "value"

    # 2. __init__
    def __init__(self, param):
        self.param = param

    # 3. Special methods (__str__, __repr__, etc.)
    def __repr__(self):
        return f"Example(param={self.param})"

    # 4. Public methods
    def public_method(self):
        """Public interface."""
        pass

    # 5. Protected methods (_method)
    def _protected_method(self):
        """Internal helper."""
        pass

    # 6. Private methods (__method)
    def __private_method(self):
        """Private implementation."""
        pass
```

## Comments

### Inline Comments

```python
# Good - Explain WHY, not WHAT
x = x + 1  # Compensate for border offset

# Avoid - States the obvious
x = x + 1  # Increment x by one

# Good - Explain complex logic
if (temperature > 0 and humidity < 60) or (pressure > 1013):
    # Only impute when conditions are within normal sensor range
    # to avoid propagating measurement errors
    result = imputer.impute(df)
```

### Block Comments

```python
# Use block comments for algorithms or complex sections

# Predictive Mean Matching (PMM) Algorithm:
# 1. Fit regression model on observed data
# 2. Predict for both observed and missing
# 3. For each missing value, find k nearest observed predictions
# 4. Randomly select one as the imputed value
for column in df.columns:
    if df[column].isna().any():
        # Implementation
        pass
```

### TODO Comments

```python
# TODO(username): Description of what needs to be done
# TODO: Add support for categorical variables
# FIXME: Memory leak when processing large datasets
# HACK: Temporary workaround for sklearn bug #12345
```

## Error Handling

### Raise Appropriate Exceptions

```python
# Good
def impute(self, df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")

    if df.empty:
        raise ValueError("Cannot impute empty DataFrame")

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    return result
```

### Use Context Managers

```python
# Good - Use context managers for resources
with open('file.txt', 'r') as f:
    data = f.read()

# Good - Custom context manager
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
```

## Best Practices

### Use List Comprehensions

```python
# Good
squared = [x**2 for x in range(10)]

# Avoid
squared = []
for x in range(10):
    squared.append(x**2)

# Good - with condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]
```

### Use f-strings

```python
# Good - f-strings (Python 3.6+)
message = f"Processing {count} rows with {missing_pct:.2%} missing"

# Avoid - old-style formatting
message = "Processing %d rows with %.2f%% missing" % (count, missing_pct)
message = "Processing {} rows".format(count)
```

### Avoid Mutable Default Arguments

```python
# Good
def function(items: Optional[list] = None) -> list:
    if items is None:
        items = []
    return items

# Avoid
def function(items: list = []) -> list:  # Dangerous!
    return items
```

### Use enumerate and zip

```python
# Good
for idx, value in enumerate(items):
    print(f"{idx}: {value}")

for x, y in zip(list1, list2):
    print(f"{x} -> {y}")

# Avoid
for idx in range(len(items)):
    value = items[idx]
    print(f"{idx}: {value}")
```

## Testing Style

```python
class TestFeature:
    """Test class for feature."""

    def test_basic_functionality(self):
        """Test basic case."""
        # Arrange
        df = pd.DataFrame({'a': [1, 2, 3]})
        expected = pd.DataFrame({'a': [1, 2, 3]})

        # Act
        result = function(df)

        # Assert
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_with_parameters(self, input, expected):
        """Test with multiple inputs."""
        assert function(input) == expected
```

## Tools

Run these tools before committing:

```bash
# Format code
poetry run black .

# Sort imports
poetry run isort .

# Lint code
poetry run flake8 imputation_showcase tests

# Type check
poetry run mypy imputation_showcase/

# Run all checks
poetry run pre-commit run --all-files
```

## Configuration Files

### pyproject.toml

```toml
[tool.black]
line-length = 88
target-version = ['py310']

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

## Next Steps

- Review [Development Setup](development.md) for environment configuration
- Check [Contributing Guidelines](guidelines.md) for the contribution process
- Start contributing with issues labeled ["good first issue"](https://github.com/DiogoRibeiro7/imputation-showcase/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

Happy coding with style! ✨
