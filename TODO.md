# TODO

This file outlines tasks to enhance the imputation-showcase repository.

- **Define and implement PMM**  
  - Write full predictive_mean_matching implementation in `src/imputation_methods.py`.  
  - Add unit tests for various missingness patterns.

- **Notebook enhancements**  
  - Load a real-world dataset (e.g., UCI housing).  
  - Visualize missing-data patterns using `missingno` or heatmaps.  
  - Illustrate each method: mean, KNN, PMM, MICE.  
  - Compare imputation results via distribution plots and error metrics.

- **Add evaluation metrics**  
  - Implement RMSE, MAE functions for imputation error.  
  - Integrate metrics into notebook and add summary table.

- **Documentation**  
  - Expand README with descriptions of each technique.  
  - Provide example code snippets in README.

- **Testing**  
  - Create a `tests/` directory.  
  - Write pytest tests for mean and KNN imputers.  
  - Test edge cases (all missing, no missing, single column).

- **CI Setup**  
  - Add GitHub Actions workflow for linting and tests.  
  - Enforce code quality with `flake8` or `pylint`.

- **Examples and visuals**  
  - Add example CSV dataset in `data/`.  
  - Generate and save plots in `figures/` for static presentation.
 
## Additional Imputation Techniques to Showcase

Add and demonstrate the following methods:

- **Median Imputation**: Replace missing values with the median of observed data.
- **Regression Imputation**: Fit regression models per feature to predict missing entries.
- **Stochastic Regression Imputation**: Add random error term to regression predictions.
- **Hot Deck Imputation**: Randomly draw donors from similar records based on categorical bins.
- **Last Observation Carried Forward (LOCF)**: For time-series data, fill with the last known value.
- **Next Observation Carried Backward (NOCB)**: Fill with the next known value in a sequence.
- **Multiple Imputation (e.g., MICE with multiple draws)**: Generate multiple complete datasets and pool results.
- **MissForest**: Random Forest–based iterative imputation for nonlinear relationships.
- **Matrix Factorization (SoftImpute)**: Low-rank matrix completion via SVD.
- **Bayesian PCA Imputation**: Probabilistic PCA to infer missing entries.
- **Autoencoder Imputation**: Use neural networks to reconstruct missing data.
- **Generative Adversarial Imputation (GAIN)**: GAN-based approach for missing data imputation.
- **Kriging / Gaussian Process Imputation**: Spatial interpolation using covariance structure.

For each technique:
- Implement the method or wrap an existing library call.
- Add usage example and comparison in the Jupyter notebook.

## notebooks/demo\_imputation.ipynb

Start an interactive notebook that:

1. Loads a synthetic or real dataset with missing values.
2. Visualizes missingness patterns.
3. Applies each imputation function.
4. Compares distributions before and after imputation.
5. Discusses pros and cons in markdown cells.
