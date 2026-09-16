# Roadmap to 1.0

<!-- --8<-- [start:roadmap] -->

`imputation-methods` 0.1.0 is the first public release. This page describes what has
to happen before 1.0.0 and the order we plan to do it in. Releases are not tied
to dates, and priorities may change based on feedback. If something here matters
to you, or is missing, please
[open an issue](https://github.com/DiogoRibeiro7/imputation-methods/issues).

## What 1.0 means

Version 1.0.0 is a promise about the public API, not a feature count. It will ship
when:

- **The API is stable.** Class names, parameter names and defaults only change
  in a major release, after a deprecation period.
- **Imputers work in machine-learning pipelines.** They can be fitted on training
  data and applied to new data, inside scikit-learn pipelines.
- **Behavior is predictable.** All imputers follow the same conventions, for
  example on errors, empty columns and output types, and each one documents its
  edge cases.
- **Statistical claims hold.** Every method does what its name and documentation
  say.

Until 1.0, minor releases (0.x) may contain breaking changes. Renamed or removed
APIs keep working for at least one minor release and raise a `FutureWarning`
that explains what to use instead.

## 0.2: Consistent API

The renames go first, so their deprecation warnings run through several releases
before 1.0 removes the old names.

- **Parameter names** *(done, unreleased)*. One name per concept: `max_iter` for
  iteration budgets, `n_neighbors` for neighbor and donor counts, `n_std` in
  `EndOfDistributionImputer`, and `strategy` for summary statistics. Old names
  work until 1.0 with a `FutureWarning`.
- **Class and function names** *(done, unreleased)*. `KNNImputerMethod` is now
  `KNNImputer`, `BayesianPCAImputer` is `PPCAImputer`, and
  `predictive_mean_matching` is `pmm_impute`.
- **Explicit fallbacks** *(done, unreleased)*. Imputers whose model can fail take
  `on_error="raise" | "fallback"`. Falling back without asking emits a
  `FutureWarning`, and the default becomes `"raise"` in 1.0, so a result never
  comes from a different method than the one the user asked for.
- **Empty columns** *(done, unreleased)*. Every imputer that learns from the data
  leaves a column with no observed values as `NaN`; only imputers that fill in a
  constant chosen by the user fill it.
- **Output types.** Define and test one dtype policy, for example whether integer
  columns stay integer after imputation.

## 0.3: scikit-learn compatibility

Today every imputer is stateless: `impute(df)` learns from and fills the same
dataframe. You can't fit on training data and then transform validation or test
data, so evaluation pipelines risk data leakage.

- Add `fit`, `transform` and `fit_transform`, keeping `impute(df)` as a shortcut
  for `fit_transform`.
- Follow scikit-learn estimator conventions (`get_params`/`set_params`, `clone`,
  parameters stored unchanged in `__init__`), so imputers work in `Pipeline`,
  `ColumnTransformer` and `GridSearchCV`, with pandas output via
  `set_output(transform="pandas")`.
- Check compliance with scikit-learn's estimator test suite.
- Document the imputers that can only fill the data they were given (for example
  carry-forward and interpolation) and how they behave in `transform`.

## 0.4: Statistical depth and evaluation

- **Multiple imputation.** `MICEImputer` returns one completed dataset. Support
  generating several imputations, and pooling estimates across them with Rubin's
  rules.
- **Expectation–maximization.** `EMImputer` runs iterative chained-equation
  imputation. Implement EM for the multivariate normal distribution, or rename the
  class to match what it does.
- **Stacking.** `StackingImputer(meta_strategy="weighted")` currently behaves like
  `"mean"`. Learn the weights on held-out observed values, or remove the option.
- **Evaluation toolkit.** Move the missingness simulators (MCAR, MAR, MNAR) from
  the notebooks into the package, add helpers that score only the masked cells,
  and add distribution-level metrics alongside RMSE and MAE.

## 0.5: Performance

- Remove the per-cell Python loop in `LocalMeanImputer`, which computes the
  distances again for every missing value.
- Speed up `KalmanFilterImputer`, whose filter recursion runs in pure Python for
  every time step of every column.
- Publish benchmark results, covering accuracy and runtime by dataset size and
  missingness pattern, in the documentation, generated from `benchmarks/`.

## 1.0.0: Stable release

- Remove the APIs deprecated during 0.x.
- Freeze the public API and follow semantic versioning strictly from here on.
- Publish versioned documentation, so each release has matching docs. The site
  currently tracks `main`.
- Distribute on conda-forge in addition to PyPI.

## After 1.0

These are worthwhile, but not needed for a stable 1.0:

- Categorical and mixed-type columns. All imputers currently require numeric
  input.
- Other dataframe libraries, such as Polars, and out-of-core or distributed data.
- GPU acceleration.

<!-- --8<-- [end:roadmap] -->
