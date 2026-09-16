"""Generate static figures for documentation from imputation demo.

This script creates visualizations comparing different imputation methods
and saves them to the repository's figures/ directory for use in documentation.

Run it from the repository root:

    python scripts/generate_figures.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes

from imputation_methods import (
    KNNImputer,
    MeanImputer,
    MedianImputer,
    MICEImputer,
    mae,
    rmse,
)

# Output directory, resolved relative to this file so the script works from any
# working directory.
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

# Create output directory
os.makedirs(FIGURES_DIR, exist_ok=True)

print("Loading diabetes dataset...")
data = load_diabetes(as_frame=True).frame
# Keep only a subset of columns for clarity
data = data.iloc[:, :5]
original = data.copy()

# Introduce 10% missing values
np.random.seed(0)
mask = np.random.rand(*data.shape) < 0.1
data[mask] = np.nan

print(f"Dataset shape: {data.shape}")
print(f"Missing values: {data.isna().sum().sum()}")

# Figure 1: Missingness pattern heatmap
print("\n1. Generating missingness pattern heatmap...")
plt.figure(figsize=(10, 6))
sns.heatmap(data.isna(), cbar=False, cmap="RdYlGn_r", yticklabels=False)
plt.title("Missing Data Pattern", fontsize=14, fontweight="bold")
plt.xlabel("Features", fontsize=12)
plt.ylabel("Samples", fontsize=12)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_missingness_pattern.png")
plt.close()
print("   Saved: figures/01_missingness_pattern.png")

# Apply imputation methods
print("\n2. Applying imputation methods...")
imputers = {
    "Mean": MeanImputer(),
    "Median": MedianImputer(),
    "KNN": KNNImputer(n_neighbors=3),
}

# Try to add more methods if they work
try:
    print("   - Mean imputation...")
    imputed_data = {"Mean": MeanImputer().impute(data)}
    print("   - Median imputation...")
    imputed_data["Median"] = MedianImputer().impute(data)
    print("   - KNN imputation...")
    imputed_data["KNN"] = KNNImputer(n_neighbors=3).impute(data)
    print("   - MICE imputation...")
    imputed_data["MICE"] = MICEImputer(random_state=0).impute(data)
except Exception as e:
    print(f"   Warning: Some methods failed: {e}")
    # Fall back to basic methods only
    imputed_data = {
        "Mean": MeanImputer().impute(data),
        "Median": MedianImputer().impute(data),
        "KNN": KNNImputer(n_neighbors=3).impute(data),
    }

# Figure 2: RMSE comparison
print("\n3. Generating RMSE comparison bar chart...")
rmse_scores = {name: rmse(original, imp_df) for name, imp_df in imputed_data.items()}
mae_scores = {name: mae(original, imp_df) for name, imp_df in imputed_data.items()}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# RMSE plot
methods = list(rmse_scores.keys())
rmse_vals = list(rmse_scores.values())
bars1 = ax1.bar(
    methods, rmse_vals, color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
)
ax1.set_ylabel("RMSE", fontsize=12)
ax1.set_title("Root Mean Squared Error by Method", fontsize=14, fontweight="bold")
ax1.set_ylim(0, max(rmse_vals) * 1.2)
for bar in bars1:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

# MAE plot
mae_vals = list(mae_scores.values())
bars2 = ax2.bar(
    methods, mae_vals, color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
)
ax2.set_ylabel("MAE", fontsize=12)
ax2.set_title("Mean Absolute Error by Method", fontsize=14, fontweight="bold")
ax2.set_ylim(0, max(mae_vals) * 1.2)
for bar in bars2:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_error_comparison.png")
plt.close()
print("   Saved: figures/02_error_comparison.png")

# Figure 3: Distribution comparison for first feature
print("\n4. Generating distribution comparison plots...")
n_methods = len(imputed_data)
n_cols = min(3, n_methods + 1)
n_rows = (n_methods + 1 + n_cols - 1) // n_cols  # ceil division
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

feature_name = data.columns[0]

# Original distribution
axes[0].hist(
    original[feature_name], bins=30, alpha=0.7, color="gray", edgecolor="black"
)
axes[0].set_title("Original (Complete)", fontsize=12, fontweight="bold")
axes[0].set_xlabel(feature_name, fontsize=10)
axes[0].set_ylabel("Frequency", fontsize=10)

# Imputed distributions
for idx, (name, imp_df) in enumerate(imputed_data.items(), start=1):
    axes[idx].hist(imp_df[feature_name], bins=30, alpha=0.7, edgecolor="black")
    axes[idx].set_title(f"{name} Imputation", fontsize=12, fontweight="bold")
    axes[idx].set_xlabel(feature_name, fontsize=10)
    axes[idx].set_ylabel("Frequency", fontsize=10)

# Hide unused subplots
for idx in range(n_methods + 1, len(axes)):
    axes[idx].axis("off")

plt.suptitle(
    f"Distribution Comparison: {feature_name}", fontsize=16, fontweight="bold", y=1.00
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_distribution_comparison.png")
plt.close()
print("   Saved: figures/03_distribution_comparison.png")

# Figure 4: Scatter plots comparing imputed vs original for missing values
print("\n5. Generating scatter plot comparison...")
n_methods = len(imputed_data)
n_cols = min(3, n_methods)
n_rows = (n_methods + n_cols - 1) // n_cols  # ceil division
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
if n_methods == 1:
    axes = np.array([axes])
elif n_rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

# Get only the positions where data was missing
missing_mask = data.isna()

for idx, (name, imp_df) in enumerate(imputed_data.items()):
    # Extract values that were imputed
    imputed_vals = imp_df[missing_mask].values.flatten()
    original_vals = original[missing_mask].values.flatten()

    # Remove any remaining NaN (shouldn't happen but just in case)
    valid_idx = ~np.isnan(imputed_vals) & ~np.isnan(original_vals)
    imputed_vals = imputed_vals[valid_idx]
    original_vals = original_vals[valid_idx]

    axes[idx].scatter(original_vals, imputed_vals, alpha=0.5, s=20)
    axes[idx].plot(
        [original_vals.min(), original_vals.max()],
        [original_vals.min(), original_vals.max()],
        "r--",
        lw=2,
        label="Perfect prediction",
    )
    axes[idx].set_xlabel("Original Value", fontsize=10)
    axes[idx].set_ylabel("Imputed Value", fontsize=10)
    axes[idx].set_title(f"{name} Imputation", fontsize=12, fontweight="bold")
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(n_methods, len(axes)):
    axes[idx].axis("off")

plt.suptitle(
    "Imputed vs Original Values (Missing Data Only)",
    fontsize=16,
    fontweight="bold",
    y=1.00,
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_scatter_comparison.png")
plt.close()
print("   Saved: figures/04_scatter_comparison.png")

# Figure 5: Summary metrics table visualization
print("\n6. Generating summary metrics table...")
summary_df = pd.DataFrame(
    {
        "Method": methods,
        "RMSE": [f"{v:.4f}" for v in rmse_vals],
        "MAE": [f"{v:.4f}" for v in mae_vals],
    }
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("tight")
ax.axis("off")

table = ax.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    cellLoc="center",
    loc="center",
    colWidths=[0.3, 0.3, 0.3],
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

# Style header
for i in range(len(summary_df.columns)):
    table[(0, i)].set_facecolor("#3498db")
    table[(0, i)].set_text_props(weight="bold", color="white")

# Alternate row colors
for i in range(1, len(summary_df) + 1):
    if i % 2 == 0:
        for j in range(len(summary_df.columns)):
            table[(i, j)].set_facecolor("#ecf0f1")

plt.title(
    "Imputation Methods Performance Summary", fontsize=14, fontweight="bold", pad=20
)
plt.savefig(FIGURES_DIR / "05_summary_table.png")
plt.close()
print("   Saved: figures/05_summary_table.png")

# Create a README for the figures directory
print("\n7. Creating figures/README.md...")
readme_content = """# Visualization Figures

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
poetry install --extras viz
poetry run python scripts/generate_figures.py
```

## Dataset

All figures are generated using the diabetes dataset from scikit-learn with 10% randomly introduced missing values.
"""

with open(FIGURES_DIR / "README.md", "w") as f:
    f.write(readme_content)
print("   Saved: figures/README.md")

print("\n" + "=" * 60)
print("All figures generated successfully!")
print("=" * 60)
print(f"\nGenerated {len(os.listdir(FIGURES_DIR)) - 1} visualization files")
print("Check the figures/ directory for output")
