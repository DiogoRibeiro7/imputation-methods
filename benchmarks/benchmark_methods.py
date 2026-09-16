"""
Performance Benchmarking Script

Benchmarks all imputation methods across various scenarios:
- Different dataset sizes
- Different missingness rates
- Different numbers of features

Generates performance reports and visualizations.
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

from imputation_methods import (
    AutoencoderImputer,
    BayesianPCAImputer,
    GAINImputer,
    GaussianProcessImputer,
    HotDeckImputer,
    KNNImputerMethod,
    LOCFImputer,
    MeanImputer,
    MedianImputer,
    MICEImputer,
    MissForestImputer,
    NOCBImputer,
    PMMImputer,
    RegressionImputer,
    SoftImputeImputer,
    StochasticRegressionImputer,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def generate_test_data(
    n_rows: int, n_cols: int, missing_rate: float, seed: int = 42
) -> tuple:
    """Generate synthetic test data with missing values.

    Args:
        n_rows: Number of rows
        n_cols: Number of columns
        missing_rate: Proportion of values to set as missing
        seed: Random seed

    Returns:
        Tuple of (complete_df, missing_df, mask)
    """
    np.random.seed(seed)

    # Generate correlated data
    mean = np.zeros(n_cols)
    # Create correlation structure
    cov = np.eye(n_cols) * 0.5 + np.ones((n_cols, n_cols)) * 0.5

    data = np.random.multivariate_normal(mean, cov, n_rows)
    df_complete = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(n_cols)])

    # Introduce missingness
    mask = np.random.rand(*df_complete.shape) < missing_rate
    df_missing = df_complete.copy()
    df_missing[mask] = np.nan

    return df_complete, df_missing, mask


def benchmark_imputer(
    imputer,
    df_missing: pd.DataFrame,
    df_complete: pd.DataFrame,
    mask: np.ndarray,
    method_name: str,
) -> dict:
    """Benchmark a single imputation method.

    Args:
        imputer: Imputer instance
        df_missing: DataFrame with missing values
        df_complete: Complete DataFrame (ground truth)
        mask: Boolean mask of missing values
        method_name: Name of the method

    Returns:
        Dictionary with benchmark results
    """
    try:
        # Time the imputation
        start_time = time.time()
        df_imputed = imputer.impute(df_missing)
        elapsed_time = time.time() - start_time

        # Calculate accuracy metrics (only on originally missing values)
        true_values = df_complete.values[mask]
        imputed_values = df_imputed.values[mask]

        # Some methods cannot fill every cell (e.g. LOCF leaves leading gaps and
        # NOCB trailing ones). Score only the cells that were filled and report
        # how many were left missing.
        filled = ~np.isnan(imputed_values)
        n_unimputed = int((~filled).sum())
        true_values = true_values[filled]
        imputed_values = imputed_values[filled]

        error_rmse = np.sqrt(np.mean((true_values - imputed_values) ** 2))
        error_mae = np.mean(np.abs(true_values - imputed_values))
        r2 = r2_score(true_values, imputed_values)

        return {
            "method": method_name,
            "time": elapsed_time,
            "rmse": error_rmse,
            "mae": error_mae,
            "r2": r2,
            "unimputed": n_unimputed,
            "success": True,
            "error": None,
        }

    except Exception as e:
        print(f"  ❌ {method_name} failed: {e!s}")
        return {
            "method": method_name,
            "time": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "unimputed": np.nan,
            "success": False,
            "error": str(e),
        }


def run_benchmark_suite(
    n_rows: int, n_cols: int, missing_rate: float, methods: dict, scenario_name: str
) -> pd.DataFrame:
    """Run benchmark on all methods for a given scenario.

    Args:
        n_rows: Number of rows
        n_cols: Number of columns
        missing_rate: Missing data rate
        methods: Dictionary of method names to imputer instances
        scenario_name: Name of the scenario

    Returns:
        DataFrame with benchmark results
    """
    print(f"\n{'=' * 80}")
    print(f"Scenario: {scenario_name}")
    print(f"  Data: {n_rows} rows × {n_cols} cols, {missing_rate:.0%} missing")
    print(f"{'=' * 80}")

    # Generate test data
    df_complete, df_missing, mask = generate_test_data(n_rows, n_cols, missing_rate)

    results = []
    for method_name, imputer in methods.items():
        print(f"  Testing {method_name:20s}...", end=" ", flush=True)

        result = benchmark_imputer(imputer, df_missing, df_complete, mask, method_name)
        result["scenario"] = scenario_name
        result["n_rows"] = n_rows
        result["n_cols"] = n_cols
        result["missing_rate"] = missing_rate

        if result["success"]:
            note = ""
            if result["unimputed"]:
                note = f"  ({result['unimputed']} of {mask.sum()} cells left missing, not scored)"
            print(f"✓ {result['time']:6.2f}s  RMSE: {result['rmse']:.4f}{note}")
        else:
            print(f"❌ {result['error']}")

        results.append(result)

    return pd.DataFrame(results)


def main():
    """Run comprehensive benchmarks."""

    print("=" * 80)
    print("IMPUTATION METHODS PERFORMANCE BENCHMARK")
    print("=" * 80)

    # Define methods to benchmark
    methods_fast = {
        "Mean": MeanImputer(),
        "Median": MedianImputer(),
        "LOCF": LOCFImputer(),
        "NOCB": NOCBImputer(),
        "KNN-3": KNNImputerMethod(k=3),
        "KNN-5": KNNImputerMethod(k=5),
        "Regression": RegressionImputer(),
        "Stochastic Reg": StochasticRegressionImputer(random_state=42),
        "PMM": PMMImputer(k=5, random_state=42),
        "Hot Deck": HotDeckImputer(random_state=42),
    }

    methods_slow = {
        "MICE": MICEImputer(random_state=42),
        "MissForest": MissForestImputer(random_state=42),
        "SoftImpute": SoftImputeImputer(max_iters=50),
        "Bayesian PCA": BayesianPCAImputer(n_components=3),
        "Autoencoder": AutoencoderImputer(
            hidden_layer_sizes=(10,), max_iter=100, random_state=42
        ),
        "GAIN": GAINImputer(random_state=42),
    }

    # Gaussian Process is very slow - only test on smallest dataset
    methods_very_slow = {
        "Gaussian Process": GaussianProcessImputer(random_state=42),
    }

    all_results = []

    # Scenario 1: Small dataset, low missingness
    results1 = run_benchmark_suite(
        n_rows=100,
        n_cols=5,
        missing_rate=0.10,
        methods={**methods_fast, **methods_slow, **methods_very_slow},
        scenario_name="Small - Low Missing",
    )
    all_results.append(results1)

    # Scenario 2: Small dataset, high missingness
    results2 = run_benchmark_suite(
        n_rows=100,
        n_cols=5,
        missing_rate=0.30,
        methods={**methods_fast, **methods_slow},
        scenario_name="Small - High Missing",
    )
    all_results.append(results2)

    # Scenario 3: Medium dataset, medium missingness
    results3 = run_benchmark_suite(
        n_rows=1000,
        n_cols=10,
        missing_rate=0.20,
        methods={**methods_fast, **methods_slow},
        scenario_name="Medium - Medium Missing",
    )
    all_results.append(results3)

    # Scenario 4: Large dataset, low missingness (only fast methods)
    results4 = run_benchmark_suite(
        n_rows=10000,
        n_cols=20,
        missing_rate=0.10,
        methods=methods_fast,
        scenario_name="Large - Low Missing",
    )
    all_results.append(results4)

    # Combine all results
    df_results = pd.concat(all_results, ignore_index=True)

    # Save results
    df_results.to_csv("benchmarks/benchmark_results.csv", index=False)
    print("\n✓ Results saved to benchmarks/benchmark_results.csv")

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(df_results)

    # Print summary
    print_summary(df_results)


def generate_visualizations(df_results: pd.DataFrame):
    """Generate benchmark visualization plots."""

    # Filter successful runs only
    df_success = df_results[df_results["success"]].copy()

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Execution time comparison (log scale)
    ax1 = fig.add_subplot(gs[0, :2])
    scenario_data = df_success[df_success["scenario"] == "Medium - Medium Missing"]
    scenario_data_sorted = scenario_data.sort_values("time")
    ax1.barh(
        scenario_data_sorted["method"], scenario_data_sorted["time"], color="steelblue"
    )
    ax1.set_xlabel("Time (seconds)", fontsize=11)
    ax1.set_title(
        "Execution Time Comparison (Medium Dataset)", fontsize=12, fontweight="bold"
    )
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3, axis="x")

    # 2. RMSE comparison
    ax2 = fig.add_subplot(gs[0, 2])
    scenario_data_rmse = scenario_data.sort_values("rmse")
    ax2.barh(scenario_data_rmse["method"], scenario_data_rmse["rmse"], color="coral")
    ax2.set_xlabel("RMSE", fontsize=10)
    ax2.set_title("Accuracy (RMSE)", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")

    # 3. Time vs Accuracy scatter
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(df_success["time"], df_success["rmse"], alpha=0.6, s=100)
    for _, row in df_success.iterrows():
        if row["time"] < 1 or row["rmse"] < 0.5:  # Label interesting points
            ax3.annotate(
                row["method"], (row["time"], row["rmse"]), fontsize=8, alpha=0.7
            )
    ax3.set_xlabel("Time (s)", fontsize=10)
    ax3.set_ylabel("RMSE", fontsize=10)
    ax3.set_title("Time vs Accuracy Tradeoff", fontsize=11, fontweight="bold")
    ax3.set_xscale("log")
    ax3.grid(True, alpha=0.3)

    # 4. Scaling with dataset size
    ax4 = fig.add_subplot(gs[1, 1])
    for method in df_success["method"].unique():
        method_data = df_success[df_success["method"] == method]
        if len(method_data) > 1:
            ax4.plot(
                method_data["n_rows"],
                method_data["time"],
                marker="o",
                label=method,
                alpha=0.7,
            )
    ax4.set_xlabel("Dataset Size (rows)", fontsize=10)
    ax4.set_ylabel("Time (s)", fontsize=10)
    ax4.set_title("Scaling Behavior", fontsize=11, fontweight="bold")
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Missing rate impact
    ax5 = fig.add_subplot(gs[1, 2])
    small_scenarios = df_success[df_success["scenario"].str.contains("Small")]
    missing_rates = (
        small_scenarios.groupby(["method", "missing_rate"])["rmse"].mean().reset_index()
    )
    for method in missing_rates["method"].unique():
        method_data = missing_rates[missing_rates["method"] == method]
        ax5.plot(
            method_data["missing_rate"], method_data["rmse"], marker="o", label=method
        )
    ax5.set_xlabel("Missing Rate", fontsize=10)
    ax5.set_ylabel("RMSE", fontsize=10)
    ax5.set_title("Impact of Missing Rate", fontsize=11, fontweight="bold")
    ax5.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Heatmap: Method x Scenario performance
    ax6 = fig.add_subplot(gs[2, :])
    pivot = df_success.pivot_table(
        values="rmse", index="method", columns="scenario", aggfunc="mean"
    )
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        ax=ax6,
        cbar_kws={"label": "RMSE"},
    )
    ax6.set_title(
        "Performance Heatmap (RMSE by Method and Scenario)",
        fontsize=12,
        fontweight="bold",
    )
    ax6.set_xlabel("Scenario", fontsize=10)
    ax6.set_ylabel("Method", fontsize=10)

    plt.savefig("benchmarks/benchmark_results.png", dpi=150, bbox_inches="tight")
    print("  ✓ Saved: benchmarks/benchmark_results.png")
    plt.close()


def print_summary(df_results: pd.DataFrame):
    """Print benchmark summary."""

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    df_success = df_results[df_results["success"]].copy()

    # Overall statistics
    print(f"\nTotal benchmarks: {len(df_results)}")
    print(
        f"Successful: {len(df_success)} ({len(df_success) / len(df_results) * 100:.1f}%)"
    )
    print(f"Failed: {len(df_results) - len(df_success)}")

    # Top 5 fastest methods
    print("\n📊 TOP 5 FASTEST METHODS (Medium Dataset):")
    medium_scenario = df_success[
        df_success["scenario"] == "Medium - Medium Missing"
    ].copy()
    top_fast = medium_scenario.nsmallest(5, "time")[["method", "time", "rmse"]]
    for _, row in top_fast.iterrows():
        print(f"  {row['method']:20s}: {row['time']:6.2f}s  (RMSE: {row['rmse']:.4f})")

    # Top 5 most accurate methods
    print("\n🎯 TOP 5 MOST ACCURATE METHODS (Medium Dataset):")
    top_accurate = medium_scenario.nsmallest(5, "rmse")[["method", "rmse", "time"]]
    for _, row in top_accurate.iterrows():
        print(f"  {row['method']:20s}: RMSE {row['rmse']:.4f}  ({row['time']:6.2f}s)")

    # Best overall (balance of speed and accuracy)
    print("\n⭐ BEST OVERALL (Speed/Accuracy Balance):")
    medium_scenario["score"] = (
        medium_scenario["time"] / medium_scenario["time"].max()
    ) + (medium_scenario["rmse"] / medium_scenario["rmse"].max())
    best_overall = medium_scenario.nsmallest(5, "score")[
        ["method", "time", "rmse", "score"]
    ]
    for _, row in best_overall.iterrows():
        print(
            f"  {row['method']:20s}: {row['time']:6.2f}s, RMSE {row['rmse']:.4f} (score: {row['score']:.3f})"
        )

    print(
        "\nRankings depend on the data. These synthetic rows have no time "
        "order, so time-series methods (LOCF, NOCB) are expected to do poorly "
        "here. See the method selection guide in the documentation."
    )

    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
