"""
Time Series Data Imputation Example

This example demonstrates how to handle missing values in time series data
using LOCF (Last Observation Carried Forward) and NOCB (Next Observation
Carried Backward) methods.

Use Case: Sensor data with intermittent missing readings
"""

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from imputation_methods import (
    KNNImputer,
    LOCFImputer,
    MeanImputer,
    NOCBImputer,
    rmse,
)

# Set plotting style
sns.set_style("whitegrid")


def generate_sensor_data(n_days=30, missing_rate=0.15, seed=42):
    """Generate synthetic sensor data and a copy with missing readings.

    Returns:
        Tuple ``(df_complete, df_missing)``: the full readings (ground truth) and
        the same frame with about ``missing_rate`` of each sensor's values set to
        NaN.
    """
    np.random.seed(seed)

    # Create timestamps
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_days * 24)]

    # Generate sensor readings with daily patterns
    hours = np.arange(len(timestamps))

    # Temperature sensor (daily cycle)
    temperature = (
        20 + 5 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 0.5, len(hours))
    )

    # Humidity sensor (inverse of temperature)
    humidity = (
        60 - 10 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 2, len(hours))
    )

    # Pressure sensor (slower cycle)
    pressure = (
        1013
        + 3 * np.sin(2 * np.pi * hours / (24 * 7))
        + np.random.normal(0, 0.3, len(hours))
    )

    # Create DataFrame
    df_complete = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
        }
    )

    # Introduce missing values (not in timestamp) in a copy, keeping the truth
    df_missing = df_complete.copy()
    for col in ["temperature", "humidity", "pressure"]:
        mask = np.random.rand(len(df_missing)) < missing_rate
        df_missing.loc[mask, col] = np.nan

    return df_complete, df_missing


def plot_comparison(original, missing, imputed_dict, column="temperature"):
    """Plot original, missing, and imputed time series."""

    fig, axes = plt.subplots(
        len(imputed_dict) + 1, 1, figsize=(14, 3 * (len(imputed_dict) + 1))
    )

    # Plot original with missing indicators
    axes[0].plot(
        original["timestamp"], original[column], "b-", label="Original", alpha=0.7
    )
    missing_mask = missing[column].isna()
    axes[0].scatter(
        missing.loc[missing_mask, "timestamp"],
        original.loc[missing_mask, column],
        color="red",
        s=50,
        label="Missing values",
        zorder=5,
    )
    axes[0].set_title(
        "Original Data with Missing Values Highlighted", fontweight="bold"
    )
    axes[0].set_ylabel(column.capitalize())
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot each imputation method
    for idx, (method_name, imputed_df) in enumerate(imputed_dict.items(), start=1):
        axes[idx].plot(
            original["timestamp"],
            original[column],
            "b--",
            label="Original",
            alpha=0.4,
            linewidth=1,
        )
        axes[idx].plot(
            imputed_df["timestamp"],
            imputed_df[column],
            "g-",
            label=f"{method_name} Imputed",
            alpha=0.8,
        )
        axes[idx].scatter(
            missing.loc[missing_mask, "timestamp"],
            imputed_df.loc[missing_mask, column],
            color="orange",
            s=30,
            label="Imputed values",
            zorder=5,
        )
        axes[idx].set_title(f"{method_name} Imputation", fontweight="bold")
        axes[idx].set_ylabel(column.capitalize())
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    return fig


def main():
    print("=" * 80)
    print("Time Series Data Imputation Example")
    print("=" * 80)

    # Generate synthetic sensor data
    print("\n1. Generating synthetic sensor data...")
    df_complete, df_missing = generate_sensor_data(n_days=7, missing_rate=0.15)

    print(f"   Dataset shape: {df_missing.shape}")
    print(
        f"   Time range: {df_missing['timestamp'].min()} to {df_missing['timestamp'].max()}"
    )
    print("   Missing values per column:")
    for col in ["temperature", "humidity", "pressure"]:
        missing_count = df_missing[col].isna().sum()
        missing_pct = missing_count / len(df_missing) * 100
        print(f"      {col:12s}: {missing_count:3d} ({missing_pct:5.2f}%)")

    # Apply different imputation methods
    print("\n2. Applying imputation methods...")

    # Prepare numeric-only data for imputation (exclude timestamp)
    numeric_cols = ["temperature", "humidity", "pressure"]
    df_numeric = df_missing[numeric_cols].copy()

    methods = {
        "LOCF": LOCFImputer(),
        "NOCB": NOCBImputer(),
        "Mean": MeanImputer(),
        "KNN": KNNImputer(n_neighbors=5),
    }

    imputed_results = {}
    for method_name, imputer in methods.items():
        print(f"   - {method_name:10s}", end=" ")
        imputed_numeric = imputer.impute(df_numeric)

        # Reconstruct full dataframe with timestamp
        imputed_full = df_missing.copy()
        imputed_full[numeric_cols] = imputed_numeric
        imputed_results[method_name] = imputed_full
        print("✓")

    # Evaluate performance on the cells that were actually missing. Scoring the
    # whole frame would dilute the error with the untouched observed values.
    print("\n3. Evaluating imputation quality (RMSE on originally missing values)...")
    print(
        f"   {'Method':<10s} {'Temp RMSE':>10s} {'Humid RMSE':>11s} {'Press RMSE':>11s}"
    )
    print("   " + "-" * 45)

    unimputed_notes = []
    for method_name, imputed_df in imputed_results.items():
        scores = []
        for col in numeric_cols:
            was_missing = df_missing[col].isna()
            true_values = df_complete.loc[was_missing, col]
            imputed_values = imputed_df.loc[was_missing, col]
            # LOCF cannot fill a leading gap and NOCB cannot fill a trailing one.
            filled = imputed_values.notna()
            if not filled.all():
                unimputed_notes.append(
                    f"{method_name} left {(~filled).sum()} {col} value(s) missing"
                )
            scores.append(rmse(true_values[filled], imputed_values[filled]))
        temp_rmse, humid_rmse, press_rmse = scores
        print(
            f"   {method_name:<10s} {temp_rmse:10.4f} {humid_rmse:11.4f} {press_rmse:11.4f}"
        )
    for note in unimputed_notes:
        print(f"   Note: {note} (not scored)")

    # Visualize results
    print("\n4. Generating visualizations...")

    # Plot temperature imputation comparison
    plot_comparison(df_complete, df_missing, imputed_results, column="temperature")
    plt.savefig(
        "examples/time_series_temperature_comparison.png", dpi=150, bbox_inches="tight"
    )
    print("   Saved: examples/time_series_temperature_comparison.png")
    plt.close()

    # Show detailed view of a short time window
    print("\n5. Analyzing specific time window...")
    window_start = 24 * 3  # Day 3
    window_end = window_start + 48  # 2 days

    _, ax = plt.subplots(figsize=(14, 6))

    # Plot data for the window
    window_slice = slice(window_start, window_end)
    times = df_complete.iloc[window_slice]["timestamp"]

    ax.plot(
        times,
        df_complete.iloc[window_slice]["temperature"],
        "ko-",
        label="Original",
        markersize=4,
        linewidth=2,
    )

    missing_in_window = df_missing.iloc[window_slice]["temperature"].isna()

    for method_name, imputed_df in imputed_results.items():
        ax.plot(
            times,
            imputed_df.iloc[window_slice]["temperature"],
            "o-",
            label=method_name,
            alpha=0.7,
            markersize=3,
        )

    # Highlight missing values
    ax.scatter(
        times[missing_in_window],
        df_complete.iloc[window_slice].loc[missing_in_window, "temperature"],
        color="red",
        s=100,
        marker="x",
        linewidths=3,
        label="Missing locations",
        zorder=10,
    )

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title("Detailed Comparison: 48-Hour Window", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("examples/time_series_detail_window.png", dpi=150, bbox_inches="tight")
    print("   Saved: examples/time_series_detail_window.png")
    plt.close()

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS FOR TIME SERIES DATA")
    print("=" * 80)
    print("""
1. LOCF (Last Observation Carried Forward):
   - Best for: Slowly changing variables (temperature, pressure)
   - Preserves temporal continuity
   - May introduce bias if missing period is long

2. NOCB (Next Observation Carried Backward):
   - Useful for backward-looking analysis
   - Complements LOCF for bidirectional filling

3. Mean Imputation:
   - Ignores temporal structure
   - Not recommended for time series with trends/cycles

4. KNN Imputation:
   - Uses rows with similar readings on the other sensors, not nearby time points
   - Works well when sensors are correlated (here temperature and humidity)
   - Computationally more expensive

RECOMMENDATION: For densely sampled sensor data, carry-forward methods (LOCF,
with NOCB for leading gaps) are a strong baseline. KNN ignores time order and
only helps for sensors that are correlated with the others.
    """)

    print("\n✓ Example complete!")


if __name__ == "__main__":
    main()
