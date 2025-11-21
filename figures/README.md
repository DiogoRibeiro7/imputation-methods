# Visualization Figures

This directory contains static visualizations generated from the imputation demo.

## Files

1. **01_missingness_pattern.png** - Heatmap showing the pattern of missing values in the dataset
2. **02_error_comparison.png** - Bar charts comparing RMSE and MAE across imputation methods
3. **03_distribution_comparison.png** - Histograms comparing feature distributions before and after imputation
4. **04_scatter_comparison.png** - Scatter plots showing imputed vs original values for missing data
5. **05_summary_table.png** - Summary table of performance metrics for all methods

## Regenerating Figures

To regenerate these figures, run:

```bash
poetry run python generate_figures.py
```

## Dataset

All figures are generated using the diabetes dataset from scikit-learn with 10% randomly introduced missing values.
