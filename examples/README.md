# Practical Examples

This directory contains practical, ready-to-run examples demonstrating real-world applications of the imputation-showcase library.

## Available Examples

### 1. Time Series Imputation (`time_series_example.py`)

**Use Case:** Sensor data with intermittent missing readings

Demonstrates handling missing values in time series data using temporal imputation methods.

**What You'll Learn:**
- When to use LOCF (Last Observation Carried Forward)
- When to use NOCB (Next Observation Carried Backward)
- Comparing temporal vs statistical methods
- Visualizing time series imputation results
- Best practices for sensor data

**Run:**
```bash
poetry run python examples/time_series_example.py
```

**Output:**
- `time_series_temperature_comparison.png` - Full comparison across methods
- `time_series_detail_window.png` - Detailed 48-hour window analysis
- Performance metrics for each method

---

### 2. Machine Learning Pipeline Integration (`ml_pipeline_example.py`)

**Use Case:** Predicting house prices with missing features

Shows how to integrate imputation into a complete machine learning workflow.

**What You'll Learn:**
- Building ML pipelines with imputation
- Preventing data leakage in train/test split
- Comparing imputation methods' impact on model performance
- Balancing accuracy vs computational cost
- Production-ready pipeline patterns

**Run:**
```bash
poetry run python examples/ml_pipeline_example.py
```

**Output:**
- `ml_pipeline_comparison.png` - Performance metrics across methods
- `ml_pipeline_predictions.png` - Actual vs predicted for best model
- Detailed evaluation metrics

**Key Components:**
- `ImputedPipeline` class - Reusable pipeline pattern
- Train/test splitting with proper imputation
- Multiple model comparisons (Ridge, Random Forest)
- Cross-validation ready structure

---

### 3. Custom Imputer Creation (`custom_imputer_example.py`)

**Use Case:** Creating domain-specific imputation methods

Demonstrates how to extend the library with your own imputation strategies.

**What You'll Learn:**
- Extending `BaseImputer` class
- Creating domain-specific imputers
- Implementing conditional imputation logic
- Combining multiple strategies
- Testing custom imputers

**Custom Imputers Included:**

1. **ModeImputer** - For discrete/categorical-like data
2. **ConditionalImputer** - Different strategies based on missingness level
3. **RobustImputer** - Trimmed mean for outlier resistance
4. **HybridImputer** - Adaptive strategy based on distribution
5. **GroupImputer** - Group-based imputation (e.g., by category)

**Run:**
```bash
poetry run python examples/custom_imputer_example.py
```

**Output:**
- `custom_imputer_comparison.png` - Performance comparison
- Detailed performance metrics
- Best practice recommendations

---

## Running All Examples

To run all examples in sequence:

```bash
# Run individually
poetry run python examples/time_series_example.py
poetry run python examples/ml_pipeline_example.py
poetry run python examples/custom_imputer_example.py

# Or create a simple script to run all
for script in examples/*.py; do
    echo "Running $script..."
    poetry run python "$script"
done
```

## Requirements

All examples require:
- Python 3.10+
- imputation-showcase package installed
- Dependencies: pandas, numpy, matplotlib, seaborn, scikit-learn

Install with:
```bash
poetry install
# or
pip install -e .
```

## Example Output

Each example generates:
- ✅ Console output with metrics and insights
- ✅ PNG visualizations saved to `examples/` directory
- ✅ Performance comparisons
- ✅ Best practice recommendations

## Understanding the Examples

### Difficulty Levels

| Example | Level | Time | Prerequisites |
|---------|-------|------|---------------|
| time_series_example.py | Intermediate | 15 min | Understanding of time series |
| ml_pipeline_example.py | Intermediate-Advanced | 20 min | Sklearn, ML concepts |
| custom_imputer_example.py | Advanced | 25 min | Python classes, OOP |

### Learning Path

```
Start → time_series_example.py (temporal methods)
        ↓
        ml_pipeline_example.py (production patterns)
        ↓
End   → custom_imputer_example.py (customization)
```

## Use Case Mapping

Choose the right example for your scenario:

**I have...**
- **Sensor/IoT data** → `time_series_example.py`
- **ML prediction task** → `ml_pipeline_example.py`
- **Unique domain needs** → `custom_imputer_example.py`

**I want to...**
- **Learn temporal imputation** → `time_series_example.py`
- **Build production pipelines** → `ml_pipeline_example.py`
- **Create custom methods** → `custom_imputer_example.py`

## Customizing Examples

All examples are designed to be easily modified:

### Modify Data
```python
# In any example, change data generation
df = generate_your_data()  # Replace with your data loading
```

### Adjust Parameters
```python
# Change imputation parameters
imputer = KNNImputerMethod(k=10)  # Increase k
```

### Add Methods
```python
# Add your own methods to comparison
methods['MyMethod'] = MyCustomImputer()
```

### Change Visualizations
```python
# Modify plots
fig, ax = plt.subplots(figsize=(16, 10))  # Larger plots
```

## Common Patterns Across Examples

All examples follow these best practices:

1. **Data Generation**
   - Realistic synthetic data
   - Controlled missing patterns
   - Ground truth for validation

2. **Evaluation**
   - Multiple metrics (RMSE, MAE, R²)
   - Visual comparisons
   - Statistical summaries

3. **Visualization**
   - High-quality figures (150 DPI)
   - Clear labels and titles
   - Saved to files for documentation

4. **Output**
   - Console progress updates
   - Key insights summary
   - Best practice recommendations

## Integration with Your Project

### Using Example Code in Production

```python
# Example: Adapting the ML pipeline pattern
from examples.ml_pipeline_example import ImputedPipeline
from imputation_showcase.imputation_methods import KNNImputerMethod
from sklearn.ensemble import GradientBoostingRegressor

# Create your pipeline
pipeline = ImputedPipeline(
    imputer=KNNImputerMethod(k=5),
    model=GradientBoostingRegressor(),
    scaler=True
)

# Train on your data
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)
```

### Creating Your Own Example

1. Copy an existing example as template
2. Modify data generation for your domain
3. Add domain-specific imputers if needed
4. Customize visualizations
5. Add to this README

## Troubleshooting

### Issue: Script Takes Too Long

**Solution:** Reduce dataset size in generation function
```python
df = generate_data(n_samples=100)  # Smaller dataset
```

### Issue: Memory Error

**Solution:** Use simpler methods or smaller data
```python
methods = {
    'Mean': MeanImputer(),  # Remove MICE, use only fast methods
    'Median': MedianImputer(),
}
```

### Issue: Plots Not Saving

**Solution:** Check directory permissions
```bash
chmod +w examples/
```

### Issue: Import Errors

**Solution:** Ensure package is installed
```bash
poetry install
# or
pip install -e .
```

## Performance Benchmarks

Approximate runtime on standard laptop (i5, 8GB RAM):

| Example | Dataset Size | Runtime | Memory |
|---------|--------------|---------|--------|
| time_series_example.py | 720 samples | ~15s | <100MB |
| ml_pipeline_example.py | 1000 samples | ~30s | <150MB |
| custom_imputer_example.py | 100 samples | ~10s | <100MB |

## Contributing Examples

Have a great real-world example? We'd love to include it!

1. Follow the existing example structure
2. Include clear documentation
3. Add visualization
4. Provide use case description
5. Submit PR (see [CONTRIBUTING.md](../CONTRIBUTING.md))

### Example Template

```python
"""
[Example Name]

Brief description of use case.

What You'll Learn:
- Key point 1
- Key point 2
"""

import relevant_libraries

def main():
    print("="*80)
    print("Example Name")
    print("="*80)

    # 1. Load/generate data
    # 2. Apply imputation
    # 3. Evaluate results
    # 4. Visualize
    # 5. Print insights

if __name__ == "__main__":
    main()
```

## Additional Resources

- **Notebooks:** [../notebooks/](../notebooks/) - Interactive tutorials
- **Documentation:** [../README.md](../README.md) - Full library documentation
- **Tests:** [../tests/](../tests/) - Unit tests showing basic usage
- **Figures:** [../figures/](../figures/) - Static visualizations

## Citation

If you use these examples in your work:

```bibtex
@software{imputation_showcase_examples,
  author = {Ribeiro, Diogo},
  title = {Imputation Showcase: Practical Examples},
  year = {2024},
  url = {https://github.com/DiogoRibeiro7/imputation-showcase}
}
```

## Support

Questions about the examples?
- Open an issue: [GitHub Issues](https://github.com/DiogoRibeiro7/imputation-showcase/issues)
- Email: [diogo.debastos.ribeiro@gmail.com](mailto:diogo.debastos.ribeiro@gmail.com)

---

**Happy Coding!** 🚀

These examples are maintained and tested with each release. Report any issues on GitHub.
