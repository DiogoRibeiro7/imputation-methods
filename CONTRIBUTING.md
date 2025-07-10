# Contributing Guidelines

Thank you for considering contributing to Imputation Showcase. This project
welcomes bug reports, feature requests, and pull requests.

## Development Setup

1. Install [Poetry](https://python-poetry.org/docs/).
2. Clone the repository and run `poetry install` to create a virtual environment
   with all dependencies.
3. Activate the shell with `poetry shell` or run commands using `poetry run`.
4. Run `flake8 imputation_showcase tests` and `pytest -q` before submitting a pull request.

## Pull Requests

- Use Google-style docstrings for all functions and classes.
- Provide inline comments where necessary for clarity.
- Ensure all tests pass and add new tests for your changes.
- Update documentation if your change modifies the public API.

## Code Style

This project uses `flake8` for linting. The CI will enforce style checks on
pushes to the `main` branch.

## Communication

Feel free to open an issue or reach out at
[diogo.debastos.ribeiro@gmail.com](mailto:diogo.debastos.ribeiro@gmail.com)
with questions about contributing.
