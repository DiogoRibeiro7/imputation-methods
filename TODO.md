# TODO

This file outlines tasks to enhance the imputation-showcase repository.

- **Define and implement PMM**
  - [x] Write full predictive_mean_matching implementation in `imputation_showcase/imputation_methods.py`.
  - [x] Add unit tests for various missingness patterns.

- **Notebook enhancements**
  - [x] Load a real-world dataset (e.g., UCI housing).
  - [x] Visualize missing-data patterns using `missingno` or heatmaps.
  - [x] Illustrate each method: mean, KNN, PMM, MICE.
  - [x] Compare imputation results via distribution plots and error metrics.

- **Add evaluation metrics**
  - [x] Implement RMSE, MAE functions for imputation error.
  - [x] Integrate metrics into notebook and add summary table.

- **Documentation**
  - [x] Expand README with descriptions of each technique.
  - [x] Provide example code snippets in README.

- **Testing**
  - [x] Create a `tests/` directory.
  - [x] Write pytest tests for mean and KNN imputers.
  - [x] Test edge cases (all missing, no missing, single column).

- **CI Setup**
  - [x] Add GitHub Actions workflow for linting and tests.
  - [x] Enforce code quality with `flake8`.

- **Examples and visuals**
  - [x] Add example CSV dataset in `data/`.
  - [ ] Generate and save plots in `figures/` for static presentation.
 
## Additional Imputation Techniques to Showcase

Add and demonstrate the following methods:

- [x] **Median Imputation**: Replace missing values with the median of observed data.
- [x] **Regression Imputation**: Fit regression models per feature to predict missing entries.
- [x] **Stochastic Regression Imputation**: Add random error term to regression predictions.
- [x] **Hot Deck Imputation**: Randomly draw donors from similar records based on categorical bins.
- [x] **Last Observation Carried Forward (LOCF)**: For time-series data, fill with the last known value.
- [x] **Next Observation Carried Backward (NOCB)**: Fill with the next known value in a sequence.
- [x] **Multiple Imputation (e.g., MICE with multiple draws)**: Generate multiple complete datasets and pool results.
- [x] **MissForest**: Random Forest–based iterative imputation for nonlinear relationships.
- [x] **Matrix Factorization (SoftImpute)**: Low-rank matrix completion via SVD.
- [x] **Bayesian PCA Imputation**: Probabilistic PCA to infer missing entries.
- [x] **Autoencoder Imputation**: Use neural networks to reconstruct missing data.
- [x] **Generative Adversarial Imputation (GAIN)**: GAN-based approach for missing data imputation.
- [x] **Kriging / Gaussian Process Imputation**: Spatial interpolation using covariance structure.

For each technique:
- Implement the method or wrap an existing library call.
- Add usage example and comparison in the Jupyter notebook.

## notebooks/demo\_imputation.ipynb

Start an interactive notebook that:

1. [x] Loads a synthetic or real dataset with missing values.
2. [x] Visualizes missingness patterns.
3. [x] Applies each imputation function.
4. [x] Compares distributions before and after imputation.
5. [x] Discusses pros and cons in markdown cells.
