# Contributing to imputation-methods

Thanks for your interest in improving imputation-methods. This guide covers how to
report problems, set up a development environment, and get a change merged.

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
Please report unacceptable behavior to
[diogo.debastos.ribeiro@gmail.com](mailto:diogo.debastos.ribeiro@gmail.com).

## Reporting bugs and requesting features

Search the [issue tracker](https://github.com/DiogoRibeiro7/imputation-methods/issues)
first, then open an issue using the bug report or feature request template. For bugs,
a minimal dataframe that reproduces the problem plus your Python, pandas, NumPy,
scikit-learn and imputation-methods versions make a fix much faster:

```bash
python -c "import sys, pandas, numpy, sklearn, imputation_methods as m; print(sys.version, pandas.__version__, numpy.__version__, sklearn.__version__, m.__version__)"
```

Security vulnerabilities should be reported privately; see [SECURITY.md](SECURITY.md).

## Development setup

You need Python 3.10+, Git and [Poetry](https://python-poetry.org/docs/#installation) 2.2+.

```bash
git clone https://github.com/<your-username>/imputation-methods.git
cd imputation-methods
poetry install                    # package + test and lint tools
poetry run pre-commit install     # run Ruff and mypy on every commit
```

Optional dependency groups: `poetry install --with docs` (MkDocs) and
`poetry install --with notebooks` (Jupyter). The plotting libraries used by the
notebooks and examples come from the `viz` extra: `poetry install --extras viz`.

Not using Poetry? Dependency groups are standard ([PEP 735](https://peps.python.org/pep-0735/)),
so `pip install -e . --group dev` (pip 25.1+) or `uv pip install -e . --group dev`
work too.

### Project layout

```text
src/imputation_methods/   the package, one module per family of imputers
  base.py                 BaseImputer, the interface every imputer implements
  functional.py           *_impute(df, ...) shortcuts
tests/                    pytest suite: one file per package module, plus
                          test_contract.py and test_functional.py, which run
                          against every imputer and every shortcut
docs/                     MkDocs site; docs/api/ is generated from docstrings
examples/, notebooks/     runnable examples (need the viz extra)
benchmarks/, scripts/     performance comparison and figure generation
```

## Checks

CI runs all of these; run them locally before opening a pull request.

| Check | Command |
| --- | --- |
| Tests (+ doctests) | `poetry run pytest` |
| Coverage (CI requires at least 90%) | `poetry run pytest --cov` |
| Lint | `poetry run ruff check .` |
| Format | `poetry run ruff format .` |
| Types (strict) | `poetry run mypy` |
| Docs | `poetry run mkdocs build --strict` (needs `--with docs`) |
| Everything pre-commit runs | `poetry run pre-commit run --all-files` |

Wall-clock performance tests are skipped by default because timings are noisy on
shared machines; run them with `poetry run pytest -m benchmark`.

CI also tests Python 3.10–3.14, Windows and macOS, and the **oldest** dependency
versions allowed by `pyproject.toml`. If you raise a minimum version, change it in
`pyproject.toml` and mention it in the changelog.

## Coding guidelines

- **Style**: Ruff formatting and linting (line length 88).
- **Types**: annotate all public functions; mypy runs in strict mode on `src/`.
- **Docstrings**: Google style, with an `Examples:` section where practical. Examples
  are executed as doctests, so keep them fast and deterministic.
- **Logging**: use the module logger with lazy formatting
  (`logger.warning("... %s", value)`). Log handled fallbacks at `WARNING`; don't
  log and then raise.
- **Randomness**: accept `random_state: int | None` and create a local
  `np.random.default_rng(random_state)`; never use the global NumPy random state.

### Adding an imputer

1. Subclass `BaseImputer` in the module for its family and implement `impute`:

    ```python
    class MyImputer(BaseImputer):
        """One-line summary.

        Longer explanation of the method and when to use it.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from imputation_methods import MyImputer
            >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
            >>> MyImputer().impute(df)["a"].tolist()
            [1.0, 2.0, 3.0]

        References:
            Author, A. (Year). Title. Venue.
        """

        def __init__(self, strength: float = 1.0) -> None:
            """Initialize the imputer.

            Args:
                strength: What this parameter controls.

            Raises:
                ValueError: If ``strength`` is negative.
            """
            if strength < 0:
                raise ValueError(f"strength must be >= 0, got {strength}")
            self.strength = strength

        def impute(self, df: pd.DataFrame) -> pd.DataFrame:
            """Fill missing values.

            Args:
                df: Numeric dataframe with missing values.

            Returns:
                A new dataframe with the same index and columns.
            """
            df = self._ensure_numeric(df)
            result = df.copy()
            ...
            return result
    ```

2. Never modify `df` in place; return a new dataframe with the same index and columns.
3. Add a `my_impute(df, ...)` shortcut to `functional.py` with the same parameters
   and defaults as the class (`tests/test_functional.py` checks this).
4. Export both from `src/imputation_methods/__init__.py` (keep `__all__` sorted).
5. Add tests in the test file for the module (e.g. `tests/test_statistical.py`).
   `tests/test_contract.py` picks up the new class automatically; add a `KWARGS`
   entry there if it can't be built with default arguments.
6. Add the class to the module's page under `docs/api/` if it's in a new module,
   to the methods table in `README.md`, and to `CHANGELOG.md`.

## Pull requests

1. Create a branch from `main` (`git switch -c fix/short-description`).
2. Keep each pull request focused on one change, with tests.
3. Add an entry under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md) for
   user-visible changes.
4. Make sure the checks above pass, then open the pull request and fill in the template.

A maintainer will review it; please respond to comments by pushing new commits
rather than force-pushing, so the review history stays readable.

## Releasing

Maintainers only. Releases are published to PyPI by
[`.github/workflows/release.yml`](.github/workflows/release.yml) using
[trusted publishing](https://docs.pypi.org/trusted-publishers/), so no API tokens
are stored in the repository.

### One-time setup

1. On PyPI, add a *pending* trusted publisher
   (<https://pypi.org/manage/account/publishing/>) for project `imputation-methods`:
   owner `DiogoRibeiro7`, repository `imputation-methods`, workflow `release.yml`,
   environment `pypi`. Do the same on TestPyPI with environment `testpypi`.
2. In the GitHub repository settings, create the environments `pypi` and
   `testpypi`. Adding yourself as a required reviewer on `pypi` gives you a manual
   approval step before anything is uploaded.
3. For the documentation site, enable GitHub Pages with source "Deploy from a
   branch", branch `gh-pages`, after the first run of the Docs workflow.

### Each release

1. Update `version` in `pyproject.toml` (following [Semantic Versioning](https://semver.org/))
   and `version`/`date-released` in `CITATION.cff`.
2. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, add a
   fresh empty `## [Unreleased]` section above it, and update the comparison links
   at the bottom.
3. Commit, open a pull request, and merge it once CI passes.
4. Optional dry run: run the **Release** workflow manually with `testpypi`, then
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ imputation-methods`.
5. Create a GitHub release with tag `vX.Y.Z` targeting `main`, using the changelog
   entry as release notes. Publishing the release builds the distributions, checks
   that the tag matches the package version, and uploads to PyPI. The **Docs**
   workflow redeploys the documentation at the same time.
