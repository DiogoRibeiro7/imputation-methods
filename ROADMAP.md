# Roadmap to 1.0

<!-- --8<-- [start:roadmap] -->

`imputation-methods` has not reached 1.0 yet. This page describes what has
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
  example on errors, empty columns, output types and invalid input, and each one
  documents its edge cases.
- **Statistical claims hold.** Every method does what its name and documentation
  say, and tests check it.
- **Support is documented.** Each release states which Python and dependency
  versions it supports.

Until 1.0, minor releases (0.x) may contain breaking changes. Renamed or removed
APIs keep working for at least one minor release and raise a `FutureWarning`
that explains what to use instead.

## At a glance

| Release | Theme | Status |
| --- | --- | --- |
| 0.2 | Consistent API | Released in 0.2.0 |
| 0.3 | scikit-learn compatibility and input validation | Next |
| 0.4 | Statistical depth and evaluation | Planned |
| 0.5 | Performance | Planned |
| 1.0.0 | Stable release | Planned |

Each milestone below lists its goal, the work, and when it counts as done. Where
a section describes current behavior, it was measured on 0.2.0.

## 0.2: Consistent API *(released in 0.2.0)*

The renames go first, so their deprecation warnings run through several releases
before 1.0 removes the old names.

- **Parameter names** *(done)*. One name per concept: `max_iter` for
  iteration budgets, `n_neighbors` for neighbor and donor counts, `n_std` in
  `EndOfDistributionImputer`, and `strategy` for summary statistics. Old names
  work until 1.0 with a `FutureWarning`.
- **Class and function names** *(done)*. `KNNImputerMethod` is now
  `KNNImputer`, `BayesianPCAImputer` is `PPCAImputer`, and
  `predictive_mean_matching` is `pmm_impute`.
- **Explicit fallbacks** *(done)*. Imputers whose model can fail take
  `on_error="raise" | "fallback"`. Falling back without asking emits a
  `FutureWarning`, and the default becomes `"raise"` in 1.0, so a result never
  comes from a different method than the one the user asked for.
- **Empty columns** *(done)*. Every imputer that learns from the data
  leaves a column with no observed values as `NaN`; only imputers that fill in a
  constant chosen by the user fill it.
- **Output types** *(done)*. Columns without missing values come back
  unchanged; imputed columns come back as floating point (`float32`/`float64` keep
  their precision, nullable columns become `Float64`).

## 0.3: scikit-learn compatibility and input validation

**Goal:** fit an imputer on one dataset, apply it to another, and use it like any
scikit-learn transformer: in `Pipeline`, `ColumnTransformer`, cross-validation and
grid search.

Today every imputer is stateless: `impute(df)` learns from and fills the same
dataframe. You can't fit on training data and then fill validation or test data,
so evaluation pipelines risk data leakage.

### Fit and transform

- Add `fit(X, y=None)`, `transform(X)` and `fit_transform(X, y=None)` to every
  imputer. `impute(df)` stays as a shortcut for `fit_transform`, and the
  functional shortcuts keep working.
- Store what `fit` learns in attributes ending in `_`, such as `statistics_`,
  `n_features_in_` and `feature_names_in_`. `transform` raises `NotFittedError`
  before `fit`, and rejects data whose columns differ from the fitted ones.
- What each family keeps from `fit`:
  - **Statistical** imputers: the per-column statistic, or per-group statistics
    for `GroupMeanImputer`.
  - **Donor sampling** imputers: the pool of observed donor values.
  - **Neighbors, regression, iterative, matrix and neural** imputers: the fitted
    models, for example per-column regressors, PPCA loadings and trained networks.
  - **Ensembles**: each fitted member.
- Time-series imputers (carry-forward, interpolation, moving averages, trends,
  seasonal and Kalman) fill a series from its own row order and have no reusable
  model. Each gets a documented `transform` behavior: either it works on the new
  data alone, or it continues from the end of the fitted series (for example,
  `LOCFImputer` carrying the last training value forward).
- Stochastic imputers draw from `random_state` in `transform` too, so results are
  reproducible.

### Estimator conventions

- Inherit from scikit-learn's `BaseEstimator` and `TransformerMixin`, for
  `get_params`/`set_params`, `clone` and the HTML representation. Ensembles expose
  their members' parameters, so a grid search can tune, for example,
  `estimator__n_neighbors` inside `BaggingImputer`.
- Keep `__init__` to storing parameters, unchanged:
  - 25 of the 41 imputers validate parameters in `__init__`, including the
    `on_error` check; validation moves to `fit`.
  - `MICEImputer`, `MissForestImputer`, `KNNImputer` and `AutoencoderImputer`
    build their scikit-learn model in `__init__`, so changing a parameter after
    construction has no effect. They will build it in `fit`.
- Rename the remaining parameters that differ from scikit-learn, with the same
  deprecation period as in 0.2: `HybridImputer(methods=...)` and
  `StackingImputer(base_imputers=...)` become `estimators=...`,
  `BaggingImputer(base_imputer=...)` becomes `estimator=...`, and
  `SoftImputeImputer(convergence_threshold=...)` becomes `tol=...`.
- Accept NumPy arrays as well as dataframes, support
  `set_output(transform="pandas")`, and implement `get_feature_names_out`,
  including the indicator columns that `IndicatorImputer` adds.
- Declare estimator tags. scikit-learn 1.6 replaced `_more_tags` with
  `__sklearn_tags__`; either support both or raise the minimum scikit-learn
  version from 1.4 to 1.6.
- Run scikit-learn's estimator checks (`parametrize_with_checks`) on every
  imputer. A check that can't apply, for example to row-order imputers, is marked
  as an expected failure with the reason.

### Input validation

Unusual input currently fails in different ways depending on the imputer:

- **Duplicate column names** crash all 41 imputers with `AttributeError`. Raise
  `ValueError` naming the duplicated columns.
- **Infinite values**: 25 imputers carry `inf` into the result, sometimes filling
  missing cells with `inf` or `NaN`. The other 16 raise either `ValueError` or
  `ImputationError`, with messages ranging from "Input contains infinity" to
  "SVD did not converge". Reject `inf` with a `ValueError` before fitting, as
  scikit-learn's own imputers do.
- **Boolean columns** work as predictors everywhere except in `LocalMeanImputer`,
  which raises `TypeError`. Treat them as 0/1 in every imputer.
- Extend the shared contract tests to cover these cases, plus zero-row dataframes,
  single rows and columns with a single observed value.

**Done when:** every imputer passes scikit-learn's estimator checks or documents
why a check doesn't apply; the machine-learning pipeline example fits on training
data and transforms test data inside `Pipeline`, `ColumnTransformer` and
`GridSearchCV`; and the contract tests cover `fit`/`transform` on separate data
and the invalid inputs above.

## 0.4: Statistical depth and evaluation

**Goal:** each method does what its name says, and users can measure which
method works for their data with tools in the package rather than notebook code.

### Multiple imputation

- `MICEImputer` is named after multiple imputation but returns one completed
  dataset, without drawing from the posterior. Add `n_imputations` to generate
  several datasets with `sample_posterior=True`, and a pooling function that
  applies Rubin's rules: the pooled estimate, within- and between-imputation
  variance, total variance, Barnard–Rubin degrees of freedom and the fraction of
  missing information.
- Add a wrapper that turns any stochastic imputer (for example `PMMImputer`,
  `HotDeckImputer` or `StochasticRegressionImputer`) into a multiple imputer by
  running it with different seeds.
- Make the draws "proper", so that pooled variances aren't understated:
  `StochasticRegressionImputer` adds residual noise to a single fitted regression,
  and `PMMImputer` matches donors on fixed coefficients. Both should draw the
  coefficients from their posterior or a bootstrap sample for each imputation.
- Expose the settings that `MICEImputer` and `MissForestImputer` hide. Today they
  accept only `random_state` and `on_error`. Add `max_iter`, `tol`, `estimator`
  and `imputation_order` to MICE, and `max_iter`, `n_estimators` and `max_depth`
  to MissForest.

### Methods that don't match their names

- **`EMImputer`** runs scikit-learn's `IterativeImputer` with `BayesianRidge`,
  which is `MICEImputer` with a tolerance. Implement expectation–maximization for
  the multivariate normal distribution (estimating the mean and covariance, then
  imputing conditional expectations), or deprecate the class in favor of
  `MICEImputer`.
- **`StackingImputer(meta_strategy="weighted")`** behaves like `"mean"`. Learn
  the weights by hiding observed cells, imputing them with each base imputer and
  weighting by accuracy, or deprecate the option.
- **`KalmanFilterImputer`** runs only the forward filter, so every value in a gap
  equals the last filtered estimate, and its noise variances are fixed parameters.
  Add a Rauch–Tung–Striebel smoother, so gaps use the observations on both sides,
  and estimate the variances from the data by maximum likelihood.
- **Docstring audit.** Check every imputer's documentation against its
  implementation, and turn each documented property into a test. Examples:
  `StochasticRegressionImputer` preserves variance, `PMMImputer` only imputes
  values that were observed, and `EndOfDistributionImputer` fills at the mean plus
  or minus `n_std` standard deviations.

### Evaluation toolkit

- **Missingness simulators.** Move `create_mcar`, `create_mar` and `create_mnar`
  from `notebooks/02_method_comparison.ipynb` into the package, with a seed, and
  return the mask alongside the data.
- **Masked scoring.** Extend `rmse` and `mae`, which currently compare two
  series, to score whole dataframes on the masked cells only, per column and
  overall.
- **Distribution metrics.** Add metrics that capture more than point accuracy:
  normalized RMSE, per-column Wasserstein distance, error in the correlation
  matrix, and for multiple imputation, the coverage of confidence intervals.
- **Method comparison.** A helper that repeatedly hides observed cells, imputes
  them with several imputers and reports the metrics, so choosing a method is
  one function call.
- **Diagnostics.** A summary of missingness per column and of missing-value
  patterns, and Little's test for data missing completely at random.

**Done when:** a worked example performs multiple imputation and pooling end to
end; every documented property of an imputer has a test; and the evaluation guide
and comparison notebook use the package's simulators and metrics.

## 0.5: Performance

**Goal:** every imputer scales predictably, and the documentation says how far.

Runtime on a laptop, for correlated data with 3 columns and 15% of the values
missing in two of them:

| Imputer | 2,000 rows | 8,000 rows | Cause |
| --- | --- | --- | --- |
| `GaussianProcessImputer` | 1.8 s | 253 s | Exact Gaussian process on every observed row |
| `MissForestImputer` | 16 s | 39 s | Random forests refitted for every column and round |
| `LocalMeanImputer` | 0.7 s | 3.7 s | Distances recomputed for every missing cell |
| `KNNImputer` | 0.04 s | 0.8 s | For comparison |

- **`GaussianProcessImputer`**: the cost grows with the cube of the number of
  rows. Cap the training rows with a `max_samples` parameter, and document the
  practical limit.
- **`LocalMeanImputer`** and **`PMMImputer`**: remove the per-cell Python loops.
  Compute neighbors for all missing cells at once, and match donors in `PMMImputer`
  by searching sorted predictions.
- **`MissForestImputer`**, **`BaggingImputer`** and **`StackingImputer`**: add
  `n_jobs` to fit members or columns in parallel.
- **Benchmarks.** `benchmarks/benchmark_methods.py` covers 16 imputers on
  synthetic Gaussian data with values missing completely at random, up to 10,000
  rows. Extend it to every imputer, to data missing at random and not at random
  (using the 0.4 simulators) and to real datasets. Publish accuracy and runtime
  tables in the documentation, generated for each release.
- **Regression tracking.** Run the tests marked `benchmark`, which are deselected
  by default, on a schedule, and flag runtime regressions.

**Done when:** no imputer loops over cells in Python; each imputer's docstring
gives its complexity or a size guideline; and the benchmark results are published.

## 1.0.0: Stable release

**Goal:** a public API that only changes in a major release.

- **Remove deprecated APIs.** This covers the old names from 0.2
  (`KNNImputerMethod`, `BayesianPCAImputer`, `predictive_mean_matching`,
  `bayesian_pca_impute`, and the parameters `max_iters`, `iterations`, `k` and
  `method` where they were renamed), and anything deprecated in 0.3 to 0.5.
- **Change the `on_error` default** to `"raise"`.
- **Freeze the public API.** The public API is the names in
  `imputation_methods.__all__`, with their signatures and defaults. A test compares
  them with a stored snapshot, so an accidental change fails CI.
- **Release candidate.** Publish `1.0.0rc1` on PyPI and collect feedback before
  the final release.
- **Versioned documentation.** Each release gets its own docs, with a version
  switcher. The site currently tracks `main`.
- **Support policy.** Document the minimum versions of Python, NumPy, pandas,
  SciPy and scikit-learn and when support for each ends, for example following the
  Scientific Python [SPEC 0](https://scientific-python.org/specs/spec-0000/).
  Today: Python 3.10 to 3.14, NumPy 1.24+, pandas 2.0+, SciPy 1.10+ and
  scikit-learn 1.4+. Python 3.10 reaches end of life in October 2026.
- **conda-forge.** Distribute on conda-forge in addition to PyPI.

## After 1.0

These are worthwhile, but not needed for a stable 1.0:

- **Categorical and mixed-type columns.** All imputers require numeric columns,
  except for the grouping column of `GroupMeanImputer`. Mode, constant, hot-deck,
  nearest-neighbor and random-forest imputation can all support categories.
- **Uncertainty for each imputed value**, for example prediction intervals from
  `KalmanFilterImputer`, `GaussianProcessImputer` and `PPCAImputer`, or the spread
  across multiple imputations.
- **Other dataframe libraries**, such as Polars, and out-of-core or distributed
  data.
- **GPU acceleration** for the neural imputers.

<!-- --8<-- [end:roadmap] -->
