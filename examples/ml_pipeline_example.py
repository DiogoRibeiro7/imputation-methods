"""
Machine Learning Pipeline Integration Example

This example demonstrates how to integrate imputation into a complete
machine learning pipeline using scikit-learn.

Use Case: Predicting house prices with missing features
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imputation_methods import (
    KNNImputer,
    MeanImputer,
    MedianImputer,
    MICEImputer,
)

sns.set_style("whitegrid")


def generate_housing_data(n_samples=1000, missing_rate=0.20):
    """Generate synthetic housing data with realistic relationships."""

    np.random.seed(42)

    # Generate features with realistic relationships
    square_feet = np.random.normal(1800, 600, n_samples)
    bedrooms = np.clip(
        np.round(square_feet / 500 + np.random.normal(0, 0.5, n_samples)), 1, 6
    )
    bathrooms = np.clip(bedrooms * 0.75 + np.random.normal(0, 0.3, n_samples), 1, 4)
    age_years = np.random.exponential(15, n_samples)
    lot_size = square_feet * np.random.uniform(1.5, 3.0, n_samples)
    garage_size = np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.5, 0.1])

    # Generate target (price) with realistic relationship
    price = (
        100000
        + 150 * square_feet
        + 20000 * bedrooms
        + 15000 * bathrooms
        - 2000 * age_years
        + 20 * lot_size
        + 10000 * garage_size
        + np.random.normal(0, 30000, n_samples)
    )

    # Create DataFrame
    df = pd.DataFrame(
        {
            "square_feet": square_feet,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age_years,
            "lot_size": lot_size,
            "garage_size": garage_size,
            "price": price,
        }
    )

    # Introduce missing values (not in target)
    feature_cols = [
        "square_feet",
        "bedrooms",
        "bathrooms",
        "age_years",
        "lot_size",
        "garage_size",
    ]
    for col in feature_cols:
        mask = np.random.rand(len(df)) < missing_rate
        df.loc[mask, col] = np.nan

    return df


class ImputedPipeline:
    """ML Pipeline with imputation step."""

    def __init__(self, imputer, model, scaler=True):
        """
        Initialize pipeline.

        Args:
            imputer: Imputation method
            model: ML model (sklearn-compatible)
            scaler: Whether to apply standardization
        """
        self.imputer = imputer
        self.model = model
        self.scaler = StandardScaler() if scaler else None

    def fit(self, X, y):
        """Fit the pipeline."""
        # Impute missing values
        X_imputed = self.imputer.impute(X)

        # Scale features
        if self.scaler:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_imputed),
                columns=X_imputed.columns,
                index=X_imputed.index,
            )
        else:
            X_scaled = X_imputed

        # Train model
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        """Make predictions."""
        # Impute missing values
        X_imputed = self.imputer.impute(X)

        # Scale features
        if self.scaler:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_imputed),
                columns=X_imputed.columns,
                index=X_imputed.index,
            )
        else:
            X_scaled = X_imputed

        # Predict
        return self.model.predict(X_scaled)


def evaluate_pipeline(pipeline, X_train, X_test, y_train, y_test, name):
    """Evaluate a pipeline and return metrics."""

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    return {
        "name": name,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
    }


def main():
    print("=" * 80)
    print("Machine Learning Pipeline Integration Example")
    print("=" * 80)

    # Generate data
    print("\n1. Generating housing dataset...")
    df = generate_housing_data(n_samples=1000, missing_rate=0.20)

    print(f"   Dataset shape: {df.shape}")
    print("   Missing values:")
    for col in df.columns:
        if col != "price":
            missing_count = df[col].isna().sum()
            missing_pct = missing_count / len(df) * 100
            print(f"      {col:15s}: {missing_count:3d} ({missing_pct:5.2f}%)")

    # Split data
    print("\n2. Splitting data into train/test sets...")
    feature_cols = [
        "square_feet",
        "bedrooms",
        "bathrooms",
        "age_years",
        "lot_size",
        "garage_size",
    ]
    X = df[feature_cols]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")

    # Define pipelines with different imputation strategies
    print("\n3. Creating ML pipelines with different imputation methods...")

    imputation_methods = {
        "Mean + Ridge": (MeanImputer(), Ridge(alpha=1.0)),
        "Median + Ridge": (MedianImputer(), Ridge(alpha=1.0)),
        "KNN + Ridge": (KNNImputer(n_neighbors=5), Ridge(alpha=1.0)),
        "MICE + Ridge": (MICEImputer(random_state=42), Ridge(alpha=1.0)),
        "KNN + RF": (
            KNNImputer(n_neighbors=5),
            RandomForestRegressor(n_estimators=100, random_state=42),
        ),
    }

    pipelines = {}
    for name, (imputer, model) in imputation_methods.items():
        pipelines[name] = ImputedPipeline(imputer, model, scaler=True)
        print(f"   ✓ {name}")

    # Evaluate all pipelines
    print("\n4. Training and evaluating pipelines...")
    results = []

    for name, pipeline in pipelines.items():
        print(f"   Evaluating {name}...", end=" ")
        result = evaluate_pipeline(pipeline, X_train, X_test, y_train, y_test, name)
        results.append(result)
        print(f"✓ Test R²: {result['test_r2']:.4f}")

    results_df = pd.DataFrame(results)

    # Display results
    print("\n5. Results Summary:")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Visualize results
    print("\n6. Generating visualizations...")

    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Test RMSE comparison
    axes[0, 0].barh(results_df["name"], results_df["test_rmse"])
    axes[0, 0].set_xlabel("Test RMSE ($)", fontsize=11)
    axes[0, 0].set_title(
        "Test Set RMSE (Lower is Better)", fontsize=12, fontweight="bold"
    )
    axes[0, 0].grid(True, alpha=0.3, axis="x")
    for i, v in enumerate(results_df["test_rmse"]):
        axes[0, 0].text(v, i, f" ${v:,.0f}", va="center", fontsize=9)

    # Test R² comparison
    axes[0, 1].barh(results_df["name"], results_df["test_r2"], color="green")
    axes[0, 1].set_xlabel("Test R² Score", fontsize=11)
    axes[0, 1].set_title(
        "Test Set R² (Higher is Better)", fontsize=12, fontweight="bold"
    )
    axes[0, 1].grid(True, alpha=0.3, axis="x")
    for i, v in enumerate(results_df["test_r2"]):
        axes[0, 1].text(v, i, f" {v:.3f}", va="center", fontsize=9)

    # Train vs Test RMSE
    x = np.arange(len(results_df))
    width = 0.35
    axes[1, 0].bar(
        x - width / 2, results_df["train_rmse"], width, label="Train", alpha=0.8
    )
    axes[1, 0].bar(
        x + width / 2, results_df["test_rmse"], width, label="Test", alpha=0.8
    )
    axes[1, 0].set_ylabel("RMSE ($)", fontsize=11)
    axes[1, 0].set_title("Train vs Test RMSE", fontsize=12, fontweight="bold")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(results_df["name"], rotation=45, ha="right")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Scatter: Train R² vs Test R²
    axes[1, 1].scatter(results_df["train_r2"], results_df["test_r2"], s=100, alpha=0.6)
    for _, row in results_df.iterrows():
        axes[1, 1].annotate(
            row["name"],
            (row["train_r2"], row["test_r2"]),
            fontsize=8,
            ha="left",
            va="bottom",
        )
    axes[1, 1].plot([0.7, 1], [0.7, 1], "r--", alpha=0.5, label="Perfect match")
    axes[1, 1].set_xlabel("Train R²", fontsize=11)
    axes[1, 1].set_ylabel("Test R²", fontsize=11)
    axes[1, 1].set_title("Overfitting Analysis", fontsize=12, fontweight="bold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("examples/ml_pipeline_comparison.png", dpi=150, bbox_inches="tight")
    print("   Saved: examples/ml_pipeline_comparison.png")
    plt.close()

    # Show predictions vs actual for best model
    best_idx = results_df["test_r2"].idxmax()
    best_name = results_df.loc[best_idx, "name"]
    best_pipeline = pipelines[best_name]

    y_pred_test = best_pipeline.predict(X_test)

    _, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_test, y_pred_test, alpha=0.5, s=30)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
        label="Perfect prediction",
    )
    ax.set_xlabel("Actual Price ($)", fontsize=12)
    ax.set_ylabel("Predicted Price ($)", fontsize=12)
    ax.set_title(
        f"Best Model: {best_name}\nTest R²: {results_df.loc[best_idx, 'test_r2']:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format axis labels as currency
    ax.ticklabel_format(style="plain", axis="both")

    plt.tight_layout()
    plt.savefig("examples/ml_pipeline_predictions.png", dpi=150, bbox_inches="tight")
    print("   Saved: examples/ml_pipeline_predictions.png")
    plt.close()

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS FOR ML PIPELINES")
    print("=" * 80)
    print(f"""
1. Best Performing Method:
   {best_name}
   - Test R²: {results_df.loc[best_idx, "test_r2"]:.4f}
   - Test RMSE: ${results_df.loc[best_idx, "test_rmse"]:,.2f}

2. Imputation Impact on Model Performance:
   - Choice of imputation method can significantly affect prediction accuracy
   - More sophisticated imputation (KNN, MICE) often improves results
   - Random Forest is more robust to imputation method choice

3. Best Practices:
   - Always split data BEFORE imputation to avoid data leakage
   - Fit imputer on training data, apply to both train and test
   - Consider imputation as a hyperparameter to tune
   - Evaluate multiple imputation strategies for your specific problem

4. Recommendations:
   - For production: Balance accuracy vs computational cost
   - For critical applications: Use MICE or KNN with cross-validation
   - For fast inference: Mean/Median may suffice if accuracy loss is acceptable
    """)

    print("\n✓ Example complete!")


if __name__ == "__main__":
    main()
