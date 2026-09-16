"""
Custom Imputer Example

This example demonstrates how to create your own custom imputation methods
by extending the BaseImputer class.

Use Cases:
- Domain-specific imputation logic
- Combining multiple strategies
- Conditional imputation based on data patterns
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from imputation_methods import BaseImputer, MeanImputer, rmse

sns.set_style("whitegrid")


# Example 1: Mode Imputer for categorical-like data
class ModeImputer(BaseImputer):
    """Impute missing values using the mode (most frequent value).

    Useful for discrete or categorical data encoded as numbers.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({'a': [1, 2, 2, np.nan, 2, 3]})
        >>> imputer = ModeImputer()
        >>> imputed = imputer.impute(df)
        >>> print(imputed.loc[3, 'a'])  # Mode is 2
        2.0
    """

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            mode_value = result[column].mode()
            if len(mode_value) > 0:
                # Use first mode if multiple exist
                result[column] = result[column].fillna(mode_value[0])
            else:
                # Fallback to mean if mode can't be computed
                result[column] = result[column].fillna(result[column].mean())

        return result


# Example 2: Conditional Imputer
class ConditionalImputer(BaseImputer):
    """Impute using different strategies based on missingness percentage.

    - Low missingness (<10%): Use mean
    - Medium missingness (10-30%): Use median
    - High missingness (>30%): Use a constant value

    Args:
        low_threshold: Threshold for low missingness (default: 0.1)
        high_threshold: Threshold for high missingness (default: 0.3)
        constant_value: Value to use for high missingness (default: 0)
    """

    def __init__(self, low_threshold=0.1, high_threshold=0.3, constant_value=0):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.constant_value = constant_value

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            missing_rate = result[column].isna().sum() / len(result)

            if missing_rate < self.low_threshold:
                # Low missingness: use mean
                fill_value = result[column].mean()
            elif missing_rate < self.high_threshold:
                # Medium missingness: use median
                fill_value = result[column].median()
            else:
                # High missingness: use constant
                fill_value = self.constant_value

            result[column] = result[column].fillna(fill_value)

        return result


# Example 3: Outlier-Robust Imputer
class RobustImputer(BaseImputer):
    """Impute using trimmed mean to avoid outlier influence.

    Uses the trimmed mean (mean after removing extreme values) which
    is more robust to outliers than standard mean.

    Args:
        trim_proportion: Proportion of values to trim from each end (default: 0.1)
    """

    def __init__(self, trim_proportion=0.1):
        self.trim_proportion = trim_proportion

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            # Calculate trimmed mean
            trimmed_mean = stats.trim_mean(
                result[column].dropna(), self.trim_proportion
            )
            result[column] = result[column].fillna(trimmed_mean)

        return result


# Example 4: Hybrid Imputer
class HybridImputer(BaseImputer):
    """Combine multiple imputation strategies.

    Uses different methods for different columns based on their
    statistical properties.

    Args:
        skewness_threshold: Threshold for determining skewed distributions (default: 1.0)
    """

    def __init__(self, skewness_threshold=1.0):
        self.skewness_threshold = skewness_threshold

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            # Calculate skewness
            skewness = result[column].skew()

            if abs(skewness) > self.skewness_threshold:
                # Skewed distribution: use median
                fill_value = result[column].median()
                strategy = "median"
            else:
                # Normal-ish distribution: use mean
                fill_value = result[column].mean()
                strategy = "mean"

            result[column] = result[column].fillna(fill_value)
            print(f"   {column:15s}: skewness={skewness:6.2f} -> using {strategy}")

        return result


# Example 5: Group-Based Imputer
class GroupImputer(BaseImputer):
    """Impute based on groups (e.g., by category).

    Note: This is a simplified example. For production, you'd need
    to handle the grouping column separately.

    Args:
        group_col: Column name to group by (must be numeric for this example)
    """

    def __init__(self, group_col):
        self.group_col = group_col

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        result = df.copy()

        # Group by the specified column and fill within groups
        for group_value in result[self.group_col].unique():
            if pd.notna(group_value):
                mask = result[self.group_col] == group_value

                for column in result.columns:
                    if column != self.group_col:
                        group_mean = result.loc[mask, column].mean()
                        result.loc[mask, column] = result.loc[mask, column].fillna(
                            group_mean
                        )

        return result


def demonstrate_custom_imputers():
    """Demonstrate all custom imputers."""

    print("=" * 80)
    print("Custom Imputer Examples")
    print("=" * 80)

    # Generate test data
    np.random.seed(42)
    n = 100

    # Create data with different characteristics
    df_complete = pd.DataFrame(
        {
            "normal": np.random.normal(50, 10, n),
            "skewed": np.random.exponential(10, n),
            "discrete": np.random.choice([1, 2, 3, 4, 5], n),
            "outliers": np.concatenate(
                [np.random.normal(100, 10, 95), [200, 250, 300, 350, 400]]
            ),
            "group": np.random.choice([1, 2, 3], n),
        }
    )

    # Introduce missing values
    df_missing = df_complete.copy()
    for col in df_missing.columns:
        if col != "group":
            mask = np.random.rand(n) < 0.15
            df_missing.loc[mask, col] = np.nan

    print("\n1. Testing Mode Imputer (for discrete data):")
    print("-" * 40)
    mode_imputer = ModeImputer()
    df_mode = mode_imputer.impute(df_missing[["discrete"]])
    print(f"   Original mode: {df_complete['discrete'].mode()[0]}")
    print(f"   Imputed with mode: {df_mode['discrete'].mode()[0]}")
    print("   ✓ Mode imputation complete")

    print("\n2. Testing Conditional Imputer:")
    print("-" * 40)
    conditional_imputer = ConditionalImputer()
    conditional_imputer.impute(df_missing[["normal", "skewed"]])
    print("   ✓ Conditional imputation complete")

    print("\n3. Testing Robust Imputer (trimmed mean):")
    print("-" * 40)
    robust_imputer = RobustImputer(trim_proportion=0.1)
    robust_imputer.impute(df_missing[["outliers"]])
    mean_value = df_complete["outliers"].mean()
    trimmed_mean_value = stats.trim_mean(df_complete["outliers"], 0.1)
    print(f"   Regular mean: {mean_value:.2f}")
    print(f"   Trimmed mean: {trimmed_mean_value:.2f}")
    print(f"   Difference: {abs(mean_value - trimmed_mean_value):.2f}")
    print("   ✓ Robust imputation less affected by outliers")

    print("\n4. Testing Hybrid Imputer (adaptive strategy):")
    print("-" * 40)
    hybrid_imputer = HybridImputer(skewness_threshold=1.0)
    hybrid_imputer.impute(df_missing[["normal", "skewed", "discrete"]])
    print("   ✓ Hybrid imputation complete")

    print("\n5. Testing Group Imputer:")
    print("-" * 40)
    group_imputer = GroupImputer(group_col="group")
    group_imputer.impute(df_missing)
    print("   ✓ Group-based imputation complete")

    # Compare all methods on normal data
    print("\n6. Performance Comparison:")
    print("=" * 80)

    methods = {
        "Mean (standard)": MeanImputer(),
        "Mode": ModeImputer(),
        "Conditional": ConditionalImputer(),
        "Robust": RobustImputer(),
        "Hybrid": HybridImputer(),
    }

    comparison_data = df_missing[["normal", "skewed"]].copy()
    comparison_complete = df_complete[["normal", "skewed"]].copy()

    results = []
    for name, imputer in methods.items():
        imputed = imputer.impute(comparison_data)
        rmse_score = rmse(comparison_complete, imputed)
        results.append({"Method": name, "RMSE": rmse_score})

    results_df = pd.DataFrame(results).sort_values("RMSE")
    print(results_df.to_string(index=False))

    # Visualize comparison
    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Original distributions
    axes[0, 0].hist(
        df_complete["normal"], bins=20, alpha=0.7, label="Normal", edgecolor="black"
    )
    axes[0, 0].set_title("Normal Distribution Data", fontweight="bold")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()

    axes[0, 1].hist(
        df_complete["skewed"],
        bins=20,
        alpha=0.7,
        color="orange",
        label="Skewed",
        edgecolor="black",
    )
    axes[0, 1].set_title("Skewed Distribution Data", fontweight="bold")
    axes[0, 1].set_xlabel("Value")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()

    # Performance comparison
    axes[1, 0].barh(results_df["Method"], results_df["RMSE"])
    axes[1, 0].set_xlabel("RMSE (Lower is Better)", fontsize=11)
    axes[1, 0].set_title("Imputation Method Comparison", fontsize=12, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis="x")
    for i, v in enumerate(results_df["RMSE"]):
        axes[1, 0].text(v, i, f" {v:.3f}", va="center", fontsize=9)

    # Outlier comparison
    mean_imputer = MeanImputer()
    robust_imputer = RobustImputer()

    df_mean_outliers = mean_imputer.impute(df_missing[["outliers"]])
    df_robust_outliers = robust_imputer.impute(df_missing[["outliers"]])

    axes[1, 1].boxplot(
        [
            df_complete["outliers"],
            df_mean_outliers["outliers"],
            df_robust_outliers["outliers"],
        ]
    )
    # Label via set_xticks: boxplot(labels=...) was removed in Matplotlib 3.11 and
    # its replacement, tick_labels=, does not exist before 3.9.
    axes[1, 1].set_xticks([1, 2, 3], ["Original", "Mean", "Robust"])
    axes[1, 1].set_title("Handling Outliers: Mean vs Robust", fontweight="bold")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("examples/custom_imputer_comparison.png", dpi=150, bbox_inches="tight")
    print("\n   Saved: examples/custom_imputer_comparison.png")
    plt.close()

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS FOR CUSTOM IMPUTERS")
    print("=" * 80)
    print("""
1. When to Create Custom Imputers:
   - Domain-specific knowledge suggests a better approach
   - Standard methods don't handle your data characteristics well
   - Need conditional logic based on data properties
   - Combining multiple strategies for different scenarios

2. Best Practices:
   - Always inherit from BaseImputer
   - Call self._ensure_numeric() to validate input
   - Document your method thoroughly
   - Test on various data patterns
   - Consider edge cases (all missing, no missing, etc.)

3. Example Use Cases:
   - Mode Imputer: Discrete/categorical numeric data
   - Conditional: Varying missingness levels
   - Robust: Data with outliers
   - Hybrid: Mixed distribution types
   - Group: Hierarchical/grouped data

4. Next Steps:
   - Extend these examples for your specific domain
   - Combine with sklearn's Pipeline for production
   - Add validation and error handling
   - Benchmark against standard methods
    """)

    print("\n✓ All custom imputer examples complete!")


if __name__ == "__main__":
    demonstrate_custom_imputers()
