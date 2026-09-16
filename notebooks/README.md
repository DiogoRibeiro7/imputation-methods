# imputation-methods Notebooks

This directory contains Jupyter notebooks demonstrating various aspects of the imputation-methods library.

## Available Notebooks

### 1. Getting Started (`01_getting_started.ipynb`)

**Level:** Beginner
**Duration:** 20-30 minutes

Learn the fundamentals of missing data imputation:
- Understanding missing data patterns
- Applying basic imputation methods (Mean, Median, KNN)
- Evaluating imputation quality
- Choosing the right method for your data

**Best for:** New users, those learning about imputation for the first time

---

### 2. Method Comparison (`02_method_comparison.ipynb`)

**Level:** Intermediate to Advanced
**Duration:** 30-45 minutes

Comparison of common imputation methods (mean, median, KNN, MICE, hot deck):
- Testing on different missing data patterns (MCAR, MAR, MNAR)
- Performance benchmarking (accuracy vs speed)
- Method selection guidelines
- Best practices for different scenarios

**Best for:** Data scientists choosing methods for production, researchers comparing approaches

---

### 3. Demo Imputation (`demo_imputation.ipynb`)

**Level:** Beginner
**Duration:** 10-15 minutes

Quick demonstration of key methods:
- Simple example with diabetes dataset
- Missingness heatmap
- RMSE comparison
- Pros and cons summary

**Best for:** Quick reference, presentations, teaching

---

## Getting Started

### Prerequisites

Ensure you have installed the package with the `viz` extra, plus Jupyter:

```bash
pip install "imputation-methods[viz]" jupyter
```

Or, from a source checkout:

```bash
poetry install --extras viz --with notebooks
```

### Running the Notebooks

1. **Start Jupyter:**
   ```bash
   poetry run jupyter notebook
   # or
   poetry run jupyter lab
   ```

2. **Navigate to** `notebooks/` directory

3. **Open** any notebook and run cells sequentially

### System Requirements

- Python 3.10+
- 4GB RAM minimum (8GB recommended for advanced notebooks)
- Jupyter Notebook or JupyterLab

## Learning Path

We recommend following this sequence:

```
1. Start Here → 01_getting_started.ipynb
                ↓
2. Then        → demo_imputation.ipynb (optional, for quick reference)
                ↓
3. Advanced    → 02_method_comparison.ipynb
```

## Notebook Features

All notebooks include:
- ✅ Clear explanations with markdown cells
- ✅ Executable code examples
- ✅ Visualizations and plots
- ✅ Performance metrics
- ✅ Best practice recommendations
- ✅ Practice exercises (where applicable)

## Common Issues and Solutions

### Issue: Kernel Dies During Execution

**Solution:** Some advanced methods (MICE, MissForest) are memory-intensive. Try:
- Restart kernel and run again
- Use smaller datasets
- Increase system RAM

### Issue: Import Errors

**Solution:** Ensure all dependencies are installed:
```bash
poetry install --extras viz --with notebooks
# or
pip install "imputation-methods[viz]" jupyter
```

If the package is installed but the import still fails, the notebook kernel is probably running a different Python environment; check with `import sys; print(sys.executable)`.

### Issue: Plotting Not Showing

**Solution:** Add this at the top of the notebook:
```python
%matplotlib inline
```

## Customizing the Notebooks

Feel free to modify the notebooks for your own use:

1. **Use your own data:** Replace the example datasets
2. **Adjust parameters:** Experiment with different imputer configurations
3. **Add methods:** Test additional imputation approaches
4. **Save results:** Export imputed data for further analysis

## Contributing

Found an issue or have a suggestion for improving the notebooks?

1. Open an issue on [GitHub](https://github.com/DiogoRibeiro7/imputation-methods/issues)
2. Submit a pull request with improvements
3. See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines

## Additional Resources

- **Documentation:** [README.md](../README.md)
- **API Reference:** [src/imputation_methods/](../src/imputation_methods/)
- **Figures:** [figures/](../figures/) - Static visualizations
- **Tests:** [tests/](../tests/) - Unit tests for reference

## Notebook Maintenance

These notebooks are tested with each release to ensure they work correctly. If you encounter any issues:

1. Check you're using the latest version
2. Update dependencies: `poetry update`
3. Report issues on GitHub

## Citation

If you use these notebooks in your research or teaching, please cite the project using the metadata in [CITATION.cff](../CITATION.cff). GitHub's "Cite this repository" button generates BibTeX and APA entries from it.

## License

These notebooks are part of the imputation-methods project and are licensed under the MIT License. See [LICENSE](../LICENSE) for details.

---

**Happy Learning!** 📚✨

For questions or support, please:
- Open an issue on [GitHub](https://github.com/DiogoRibeiro7/imputation-methods/issues)
- Email: [diogo.debastos.ribeiro@gmail.com](mailto:diogo.debastos.ribeiro@gmail.com)
