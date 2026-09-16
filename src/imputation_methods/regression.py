"""Regression-based imputers that predict each column from the others."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import (
    BayesianRidge,
    HuberRegressor,
    LinearRegression,
    RANSACRegressor,
)

from ._deprecation import renamed_parameters
from .base import BaseImputer

logger = logging.getLogger(__name__)


def _predictor_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the predictors used to model each incomplete column.

    Gaps in the predictors are filled with column means so that rows with
    several missing values can still be used. Columns with no observed values
    carry no information and are dropped.
    """
    usable = df.loc[:, df.notna().any()]
    return usable.fillna(usable.mean())


class RegressionImputer(BaseImputer):
    """Impute missing values via linear regression.

    Each incomplete column is regressed on all other columns, using the rows
    where it is observed. Gaps in the predictor columns are mean-filled first.
    """

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict missing entries using other columns as features.

        Args:
            df: Dataframe with potential NaN values.

        Returns:
            Imputed dataframe with missing values filled by regression
            predictions.
        """
        df = self._ensure_numeric(df)
        result = df.copy()
        features = _predictor_frame(df)

        for column in result.columns:
            if result[column].isna().any():
                predictors = features.columns.difference([column])
                observed_mask = result[column].notna().to_numpy()
                observed = result[observed_mask]
                x_observed = features.loc[observed_mask, predictors]
                x_missing = features.loc[~observed_mask, predictors]
                missing = result[~observed_mask]

                if predictors.size == 0 or observed.empty:
                    logger.warning(
                        "No predictors or observations for column %r; skipping",
                        column,
                    )
                    continue

                try:
                    reg = LinearRegression()
                    reg.fit(x_observed, observed[column])
                    predicted = reg.predict(x_missing)
                    result.loc[missing.index, column] = predicted
                except (ValueError, np.linalg.LinAlgError) as e:
                    logger.warning(
                        "Regression failed for column %r (%s); using mean imputation",
                        column,
                        e,
                    )
                    result[column] = result[column].fillna(result[column].mean())
                except Exception as e:
                    logger.warning(
                        "Unexpected error in regression for column %r (%s); "
                        "using median imputation",
                        column,
                        e,
                    )
                    result[column] = result[column].fillna(result[column].median())

        return result


class StochasticRegressionImputer(BaseImputer):
    """Impute missing values with regression plus random noise.

    Like :class:`RegressionImputer`, but adds Gaussian noise with the standard
    deviation of the regression residuals, which preserves the variance of the
    imputed column instead of shrinking it towards the regression line.
    """

    def __init__(self, random_state: int | None = None) -> None:
        """Initialize the imputer.

        Args:
            random_state: Seed controlling the noise generation.
        """
        self.random_state = random_state

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict missing entries and add Gaussian noise.

        Args:
            df: Dataframe with potential NaN values.

        Returns:
            Imputed dataframe with stochastic regression predictions.
        """
        rng = np.random.default_rng(self.random_state)
        df = self._ensure_numeric(df)
        result = df.copy()
        features = _predictor_frame(df)

        for column in result.columns:
            if result[column].isna().any():
                predictors = features.columns.difference([column])
                observed_mask = result[column].notna().to_numpy()
                observed = result[observed_mask]
                x_observed = features.loc[observed_mask, predictors]
                x_missing = features.loc[~observed_mask, predictors]
                missing = result[~observed_mask]

                if predictors.size == 0 or observed.empty:
                    continue

                reg = LinearRegression()
                reg.fit(x_observed, observed[column])
                predicted = reg.predict(x_missing)
                # fmt: off
                residuals = observed[column] - reg.predict(
                    x_observed
                )
                # fmt: on
                std = residuals.std(ddof=0)
                noise = rng.normal(0, std, size=predicted.shape)
                result.loc[missing.index, column] = predicted + noise

        return result


class PMMImputer(BaseImputer):
    """Impute missing values using predictive mean matching (PMM).

    For each column, a linear regression on the other columns scores every row.
    Each missing entry is then replaced by the observed value of a donor drawn
    at random from the ``n_neighbors`` rows whose predicted scores are closest, so
    imputed values are always values that actually occur in the data.

    References:
        Little, R. J. A. (1988). Missing-data adjustments in large surveys.
        Journal of Business & Economic Statistics, 6(3), 287-296.
    """

    @renamed_parameters(k="n_neighbors")
    def __init__(self, n_neighbors: int = 5, random_state: int | None = None) -> None:
        """Initialize the imputer.

        Args:
            n_neighbors: Number of donor candidates to consider.
            random_state: Seed for donor selection randomness.

        Raises:
            ValueError: If n_neighbors is not a positive integer.
        """
        if not isinstance(n_neighbors, int):
            raise TypeError(
                f"n_neighbors must be an integer, got {type(n_neighbors).__name__}"
            )
        if n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be positive, got {n_neighbors}")
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute data using predictive mean matching.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe. Columns with no observed values are left as-is.
        """
        rng = np.random.default_rng(self.random_state)
        df = self._ensure_numeric(df)
        result = df.copy()
        features = _predictor_frame(df)

        for column in result.columns:
            if result[column].isna().any():
                predictors = features.columns.difference([column])
                observed_mask = result[column].notna().to_numpy()
                observed = result[observed_mask]
                x_observed = features.loc[observed_mask, predictors]
                x_missing = features.loc[~observed_mask, predictors]
                missing = result[~observed_mask]

                if observed.empty:
                    # Nothing to match against; leave the column untouched.
                    continue
                if predictors.size == 0:
                    result[column] = result[column].fillna(observed[column].mean())
                    continue

                try:
                    reg = LinearRegression()
                    reg.fit(x_observed, observed[column])

                    observed_pred = reg.predict(x_observed)
                    missing_pred = reg.predict(x_missing)
                    observed_values = observed[column].to_numpy()

                    for i, pred in zip(missing.index, missing_pred, strict=True):
                        distances = np.abs(observed_pred - pred)
                        nearest_idx = np.argsort(distances)[: self.n_neighbors]
                        result.at[i, column] = rng.choice(observed_values[nearest_idx])
                except (ValueError, np.linalg.LinAlgError) as e:
                    logger.warning(
                        "PMM failed for column %r (%s); using mean imputation",
                        column,
                        e,
                    )
                    result[column] = result[column].fillna(result[column].mean())
                except Exception as e:
                    logger.warning(
                        "Unexpected error in PMM for column %r (%s); "
                        "using median imputation",
                        column,
                        e,
                    )
                    result[column] = result[column].fillna(result[column].median())

        return result


class BayesianRidgeImputer(BaseImputer):
    """Bayesian ridge regression imputation for missing values.

    Uses Bayesian ridge regression to predict missing values based on other
    features. Provides probabilistic estimates and handles multicollinearity well.

    **Algorithm Overview:**
    Bayesian ridge regression treats the regression coefficients as random variables
    with Gaussian priors. It iteratively estimates both the coefficients and the
    precision (inverse variance) parameters using an Expectation-Maximization
    approach. This provides automatic relevance determination - features with
    low relevance are automatically down-weighted.

    **Key Differences from Standard Ridge Regression:**
    - Standard ridge: Fixed regularization parameter λ (must be tuned)
    - Bayesian ridge: Learns optimal α (precision of weights) and λ (precision of noise)
    - Provides uncertainty estimates through posterior distributions
    - More robust to overfitting with automatic parameter adaptation

    **Hyperparameter Priors:**
    - alpha ~ Gamma(alpha_1, alpha_2): Controls precision of weights (inverse variance)
    - lambda ~ Gamma(lambda_1, lambda_2): Controls precision of noise
    - Small values (1e-6) create weak priors, letting data dominate
    - Larger values create stronger priors, enforcing more regularization

    **When to Use:**
    - Small to medium datasets with multivariate relationships
    - When feature relevance is unknown (automatic feature selection)
    - When uncertainty quantification is valuable
    - When you want to avoid manual hyperparameter tuning

    Args:
        max_iter: Maximum iterations for optimization. Default: 300
        tol: Convergence tolerance. Default: 1e-3
        alpha_1: Hyper-parameter for Gamma prior over alpha. Default: 1e-6
        alpha_2: Hyper-parameter for Gamma prior over alpha. Default: 1e-6
        lambda_1: Hyper-parameter for Gamma prior over lambda. Default: 1e-6
        lambda_2: Hyper-parameter for Gamma prior over lambda. Default: 1e-6

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_methods import BayesianRidgeImputer
        >>> df = pd.DataFrame({
        ...     'a': [1, 2, np.nan, 4, 5],
        ...     'b': [2, 4, 6, np.nan, 10]
        ... })
        >>> imputer = BayesianRidgeImputer()
        >>> imputed = imputer.impute(df)

    References:
        Bayesian approach to ridge regression with automatic relevance determination.
        MacKay, D. J. C. (1992). Bayesian interpolation.
    """

    def __init__(
        self,
        max_iter: int = 300,
        tol: float = 1e-3,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
    ) -> None:
        """Initialize the Bayesian ridge imputer.

        Args:
            max_iter: Maximum iterations
            tol: Convergence tolerance
            alpha_1: Shape parameter of the Gamma prior over ``alpha``.
            alpha_2: Rate parameter of the Gamma prior over ``alpha``.
            lambda_1: Shape parameter of the Gamma prior over ``lambda``.
            lambda_2: Rate parameter of the Gamma prior over ``lambda``.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using Bayesian ridge regression.

        Implements a column-by-column imputation strategy where each column
        with missing values is predicted using all other columns as features.
        The Bayesian ridge model automatically learns optimal regularization
        parameters during the iterative fitting process.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        # Process each column independently, using the others as predictors
        for column in result.columns:
            if result[column].isna().any():
                # Split data into training (observed) and prediction (missing) sets
                # This is crucial for supervised learning approach to imputation
                train_mask = ~result[column].isna()
                predict_mask = result[column].isna()

                # Edge case: No observed values to learn from
                if train_mask.sum() == 0:
                    # Fallback to zero (could also use global mean)
                    result[column] = result[column].fillna(0)
                    continue

                # Use all other columns as predictive features
                # This exploits multivariate relationships in the data
                feature_cols = [c for c in result.columns if c != column]
                if len(feature_cols) == 0:
                    # Edge case: Single-column dataframe, use univariate mean
                    result[column] = result[column].fillna(result[column].mean())
                    continue

                # Prepare training and prediction matrices
                # Note: fillna(0) for features is a simple strategy; could be improved
                # with more sophisticated feature imputation
                X_train = result.loc[train_mask, feature_cols].fillna(0).values
                y_train = result.loc[train_mask, column].values
                X_predict = result.loc[predict_mask, feature_cols].fillna(0).values

                if len(X_train) > 0 and len(X_predict) > 0:
                    # Initialize Bayesian ridge model with specified priors
                    # The model will iteratively update α (weight precision) and
                    # λ (noise precision) to find optimal posterior distributions
                    model = BayesianRidge(
                        max_iter=self.max_iter,  # Usually converges in < 100 iterations
                        tol=self.tol,  # Stop when change in log-likelihood < tol
                        alpha_1=self.alpha_1,  # Shape parameter for alpha prior
                        alpha_2=self.alpha_2,  # Rate parameter for alpha prior
                        lambda_1=self.lambda_1,  # Shape parameter for lambda prior
                        lambda_2=self.lambda_2,  # Rate parameter for lambda prior
                    )

                    # Fit: Iteratively update coefficients w, α, and λ
                    # Uses conjugate Gaussian-Gamma priors for analytical updates
                    model.fit(X_train, y_train)

                    # Predict: Returns posterior mean estimates (point predictions)
                    # Could also return std via predict with return_std=True
                    predictions = model.predict(X_predict)
                    result.loc[predict_mask, column] = predictions

        return result


class HuberImputer(BaseImputer):
    """Robust regression imputation using Huber loss.

    Uses Huber regression which is robust to outliers in both
    features and target values.

    Args:
        epsilon: Huber loss parameter (controls outlier threshold). Default: 1.35
        max_iter: Maximum iterations. Default: 100
        alpha: Regularization strength. Default: 0.0001

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_methods import HuberImputer
        >>> df = pd.DataFrame({
        ...     'a': [1, 2, np.nan, 100, 5],  # 100 is outlier
        ...     'b': [2, 4, 6, 200, np.nan]
        ... })
        >>> imputer = HuberImputer()
        >>> imputed = imputer.impute(df)

    References:
        Huber, P. J. (1964). Robust estimation of a location parameter.
    """

    def __init__(
        self, epsilon: float = 1.35, max_iter: int = 100, alpha: float = 0.0001
    ) -> None:
        """Initialize the Huber imputer.

        Args:
            epsilon: Huber loss parameter
            max_iter: Maximum iterations
            alpha: Regularization strength
        """
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.alpha = alpha

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using Huber regression.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            if result[column].isna().any():
                train_mask = ~result[column].isna()
                predict_mask = result[column].isna()

                if train_mask.sum() == 0:
                    result[column] = result[column].fillna(0)
                    continue

                feature_cols = [c for c in result.columns if c != column]
                if len(feature_cols) == 0:
                    result[column] = result[column].fillna(result[column].mean())
                    continue

                X_train = result.loc[train_mask, feature_cols].fillna(0).values
                y_train = result.loc[train_mask, column].values
                X_predict = result.loc[predict_mask, feature_cols].fillna(0).values

                if len(X_train) > 1 and len(X_predict) > 0:
                    model = HuberRegressor(
                        epsilon=self.epsilon, max_iter=self.max_iter, alpha=self.alpha
                    )
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_predict)
                    result.loc[predict_mask, column] = predictions

        return result


class RANSACImputer(BaseImputer):
    """RANSAC robust regression for outlier-resistant imputation.

    Uses RANSAC (Random Sample Consensus) to fit regression models
    that are robust to outliers in the training data.

    Args:
        min_samples: Minimum samples for model. Default: None (auto)
        residual_threshold: Threshold for inliers. Default: None (auto)
        max_trials: Maximum RANSAC iterations. Default: 100
        random_state: Random seed. Default: None

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_methods import RANSACImputer
        >>> df = pd.DataFrame({
        ...     'a': [1, 2, np.nan, 100, 5],  # 100 is outlier
        ...     'b': [2, 4, 6, 200, np.nan]
        ... })
        >>> imputer = RANSACImputer()
        >>> imputed = imputer.impute(df)

    References:
        Fischler & Bolles (1981). Random sample consensus.
    """

    def __init__(
        self,
        min_samples: int | None = None,
        residual_threshold: float | None = None,
        max_trials: int = 100,
        random_state: int | None = None,
    ) -> None:
        """Initialize the RANSAC imputer.

        Args:
            min_samples: Minimum samples for model
            residual_threshold: Inlier threshold
            max_trials: Maximum iterations
            random_state: Random seed
        """
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.random_state = random_state

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using RANSAC regression.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            if result[column].isna().any():
                train_mask = ~result[column].isna()
                predict_mask = result[column].isna()

                if train_mask.sum() < 3:  # RANSAC needs at least 3 samples
                    result[column] = result[column].fillna(result[column].mean())
                    continue

                feature_cols = [c for c in result.columns if c != column]
                if len(feature_cols) == 0:
                    result[column] = result[column].fillna(result[column].mean())
                    continue

                X_train = result.loc[train_mask, feature_cols].fillna(0).values
                y_train = result.loc[train_mask, column].values
                X_predict = result.loc[predict_mask, feature_cols].fillna(0).values

                if len(X_predict) > 0:
                    try:
                        model = RANSACRegressor(
                            min_samples=self.min_samples,
                            residual_threshold=self.residual_threshold,
                            max_trials=self.max_trials,
                            random_state=self.random_state,
                        )
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_predict)
                        result.loc[predict_mask, column] = predictions
                    except Exception:
                        # Fall back to median if RANSAC fails
                        result.loc[predict_mask, column] = result[column].median()

        return result


class GaussianProcessImputer(BaseImputer):
    """Impute missing values using Gaussian Process regression."""

    def __init__(
        self,
        kernel: RBF | None = None,
        alpha: float = 1e-10,
        random_state: int | None = None,
    ) -> None:
        """Initialize the imputer.

        Args:
            kernel: Kernel used by the Gaussian process. Defaults to ``RBF``.
            alpha: Added noise term to ensure numerical stability.
            random_state: Random seed for reproducibility.
        """
        self.kernel = kernel or RBF()
        self.alpha = alpha
        self.random_state = random_state

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict missing entries with a Gaussian Process model.

        Args:
            df: Dataframe containing missing values.

        Returns:
            Dataframe where NaNs are replaced by GP predictions.
        """
        df = self._ensure_numeric(df)
        result = df.copy()
        features = _predictor_frame(df)

        for column in result.columns:
            if result[column].isna().any():
                predictors = features.columns.difference([column])
                observed_mask = result[column].notna().to_numpy()
                observed = result[observed_mask]
                x_observed = features.loc[observed_mask, predictors]
                x_missing = features.loc[~observed_mask, predictors]
                missing = result[~observed_mask]

                if predictors.size == 0 or observed.empty:
                    logger.warning(
                        "No predictors or observations for column %r; skipping",
                        column,
                    )
                    continue

                try:
                    gp = GaussianProcessRegressor(
                        kernel=self.kernel,
                        alpha=self.alpha,
                        random_state=self.random_state,
                    )
                    gp.fit(x_observed, observed[column])
                    predicted = gp.predict(x_missing)
                    result.loc[missing.index, column] = predicted
                except (ValueError, np.linalg.LinAlgError) as e:
                    logger.warning(
                        "Gaussian process failed for column %r (%s); "
                        "using mean imputation",
                        column,
                        e,
                    )
                    result[column] = result[column].fillna(result[column].mean())
                except Exception as e:
                    logger.warning(
                        "Unexpected error in Gaussian process for column %r (%s); "
                        "using median imputation",
                        column,
                        e,
                    )
                    result[column] = result[column].fillna(result[column].median())

        return result
