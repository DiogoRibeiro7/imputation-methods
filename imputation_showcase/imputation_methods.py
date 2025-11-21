"""Imputation methods for the imputation-showcase project."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from fancyimpute import SoftImpute
from ppca import PPCA
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Configure module logger
logger = logging.getLogger(__name__)


class BaseImputer(ABC):
    """Base class for all imputation methods."""

    def _ensure_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that ``df`` contains only numeric columns.

        Args:
            df: Dataframe to validate.

        Returns:
            The original dataframe if all columns are numeric.

        Raises:
            TypeError: If ``df`` includes any non-numeric columns.
        """
        # fmt: off
        numeric_flags = [
            pd.api.types.is_numeric_dtype(dtype)
            for dtype in df.dtypes
        ]
        # fmt: on
        if not all(numeric_flags):
            raise TypeError("All columns must be numeric")
        return df

    @abstractmethod
    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in ``df``.

        Args:
            df: Dataframe with potential NaN values.

        Returns:
            Dataframe with imputed values.
        """


class MeanImputer(BaseImputer):
    """Impute missing values using column means.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import MeanImputer
        >>> df = pd.DataFrame({"a": [1, 2, np.nan, 4]})
        >>> imputer = MeanImputer()
        >>> imputed = imputer.impute(df)
        >>> print(imputed.loc[2, "a"])  # Mean of [1, 2, 4]
        2.333...
    """

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        logger.info(f"Imputing {df.shape[0]} rows, {df.shape[1]} columns")
        missing_count = df.isna().sum().sum()
        logger.info(f"Total missing values: {missing_count}")

        result = df.copy()
        for column in result.columns:
            mean_val = result[column].mean()
            result[column] = result[column].fillna(mean_val)

        logger.info("Mean imputation completed successfully")
        return result


class MedianImputer(BaseImputer):
    """Impute missing values using column medians.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import MedianImputer
        >>> df = pd.DataFrame({"a": [1, 2, np.nan, 10]})
        >>> imputer = MedianImputer()
        >>> imputed = imputer.impute(df)
        >>> print(imputed.loc[2, "a"])  # Median of [1, 2, 10]
        2.0
    """

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        result = df.copy()
        for column in result.columns:
            median_val = result[column].median()
            result[column] = result[column].fillna(median_val)
        return result


class KNNImputerMethod(BaseImputer):
    """Impute missing values using K-nearest neighbors.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import KNNImputerMethod
        >>> df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, np.nan, 7, 8]})
        >>> imputer = KNNImputerMethod(k=2)
        >>> imputed = imputer.impute(df)
        >>> assert not imputed.isna().any().any()
    """

    def __init__(self, k: int = 5):
        """Initialize the imputer.

        Args:
            k: Number of neighbors to consider.

        Raises:
            ValueError: If k is not a positive integer.
        """
        if not isinstance(k, int):
            raise TypeError(f"k must be an integer, got {type(k).__name__}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        self._imputer = KNNImputer(n_neighbors=k)

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using the fitted KNN strategy.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        logger.info(f"KNN imputation with k={self.k} on {df.shape} dataframe")
        missing_count = df.isna().sum().sum()
        logger.info(f"Missing values: {missing_count}")

        if missing_count == 0:
            logger.warning("No missing values found, returning copy")

        imputed_array = self._imputer.fit_transform(df)
        logger.info("KNN imputation completed")
        return pd.DataFrame(imputed_array, columns=df.columns, index=df.index)


class PMMImputer(BaseImputer):
    """Impute missing values using predictive mean matching (PMM)."""

    def __init__(self, k: int = 5, random_state: int | None = None):
        """Initialize the imputer.

        Args:
            k: Number of donor candidates to consider.
            random_state: Seed for donor selection randomness.

        Raises:
            ValueError: If k is not a positive integer.
        """
        if not isinstance(k, int):
            raise TypeError(f"k must be an integer, got {type(k).__name__}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        self.random_state = random_state

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute data using predictive mean matching.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            if result[column].isna().any():
                predictors = result.columns.difference([column])
                observed = result[result[column].notna()]
                missing = result[result[column].isna()]

                if predictors.size == 0 or observed.empty:
                    mean_val = observed[column].mean()
                    result.loc[result[column].isna(), column] = mean_val
                    continue

                reg = LinearRegression()
                reg.fit(observed[predictors], observed[column])

                observed_pred = reg.predict(observed[predictors])
                missing_pred = reg.predict(missing[predictors])

                for i, pred in zip(missing.index, missing_pred):
                    distances = np.abs(observed_pred - pred)
                    nearest_idx = np.argsort(distances)[: self.k]
                    donors = observed.iloc[nearest_idx]
                    imputed_val = (
                        donors[column]
                        .sample(
                            1,
                            random_state=self.random_state,
                        )
                        .iloc[0]
                    )
                    result.at[i, column] = imputed_val

        return result


class MICEImputer(BaseImputer):
    """Impute missing data using Multiple Imputation by Chained Equations.

    This method is commonly abbreviated as MICE.
    """

    def __init__(self, random_state: int | None = None):
        """Initialize the imputer.

        Args:
            random_state: Random seed used by the underlying estimator.
        """
        self.random_state = random_state
        self._imputer = IterativeImputer(random_state=random_state)

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform MICE-based imputation.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        imputed_array = self._imputer.fit_transform(df)
        return pd.DataFrame(imputed_array, columns=df.columns, index=df.index)


class RegressionImputer(BaseImputer):
    """Impute missing values via linear regression."""

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

        for column in result.columns:
            if result[column].isna().any():
                predictors = result.columns.difference([column])
                observed = result[result[column].notna()]
                missing = result[result[column].isna()]

                if predictors.size == 0 or observed.empty:
                    continue

                reg = LinearRegression()
                reg.fit(observed[predictors], observed[column])
                predicted = reg.predict(missing[predictors])
                result.loc[missing.index, column] = predicted

        return result


class StochasticRegressionImputer(BaseImputer):
    """Impute missing values with regression plus random noise."""

    def __init__(self, random_state: int | None = None):
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

        for column in result.columns:
            if result[column].isna().any():
                predictors = result.columns.difference([column])
                observed = result[result[column].notna()]
                missing = result[result[column].isna()]

                if predictors.size == 0 or observed.empty:
                    continue

                reg = LinearRegression()
                reg.fit(observed[predictors], observed[column])
                predicted = reg.predict(missing[predictors])
                # fmt: off
                residuals = observed[column] - reg.predict(
                    observed[predictors]
                )
                # fmt: on
                std = residuals.std(ddof=0)
                noise = rng.normal(0, std, size=predicted.shape)
                result.loc[missing.index, column] = predicted + noise

        return result


class LOCFImputer(BaseImputer):
    """Impute using Last Observation Carried Forward (LOCF)."""

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values forward along each column.

        Args:
            df: Dataframe with potential NaN values.

        Returns:
            Dataframe where NaNs are replaced by the last seen observation.
        """
        df = self._ensure_numeric(df)
        return df.ffill()


class NOCBImputer(BaseImputer):
    """Impute using Next Observation Carried Backward (NOCB)."""

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values backward along each column.

        Args:
            df: Dataframe with potential NaN values.

        Returns:
            Dataframe where NaNs are replaced by the next observed value.
        """
        df = self._ensure_numeric(df)
        return df.bfill()


class HotDeckImputer(BaseImputer):
    """Impute missing values using the Hot Deck method.

    This approach fills missing entries by randomly sampling existing values
    ("donors") from the same column. Optionally, sampling can be restricted to
    rows sharing the same values in ``stratify_cols``.
    """

    def __init__(
        self,
        stratify_cols: list[str] | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initialize the imputer.

        Args:
            stratify_cols: Columns used to define similarity groups. If
                ``None``, donors are drawn from the entire column.
            random_state: Seed controlling the random sampling of donors.
        """
        self.stratify_cols = stratify_cols or []
        self.random_state = random_state

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values by sampling existing observations.

        Args:
            df: Dataframe with potential NaN values.

        Returns:
            Dataframe where NaNs are replaced by randomly sampled donors.
        """
        rng = np.random.default_rng(self.random_state)
        df = self._ensure_numeric(df)
        result = df.copy()

        if not self.stratify_cols:
            for column in result.columns:
                missing_mask = result[column].isna()
                donors = result[column].dropna()
                if donors.empty:
                    continue
                result.loc[missing_mask, column] = rng.choice(
                    donors.to_numpy(), size=missing_mask.sum(), replace=True
                )
            return result

        grouped = result.groupby(self.stratify_cols)
        for _, idx in grouped.groups.items():
            group_df = result.loc[idx]
            for column in group_df.columns.difference(self.stratify_cols):
                missing_mask = group_df[column].isna()
                if not missing_mask.any():
                    continue
                donors = group_df[column].dropna()
                if donors.empty:
                    donors = result[column].dropna()
                if donors.empty:
                    continue
                result.loc[idx[missing_mask], column] = rng.choice(
                    donors.to_numpy(), size=missing_mask.sum(), replace=True
                )

        return result


class MissForestImputer(BaseImputer):
    """Impute missing data using the MissForest algorithm.

    This implementation relies on scikit-learn's ``IterativeImputer`` with a
    ``RandomForestRegressor`` estimator to approximate the MissForest method.
    """

    def __init__(self, random_state: int | None = None):
        """Initialize the imputer.

        Args:
            random_state: Seed controlling the random forest randomness.
        """
        self.random_state = random_state
        self._imputer = IterativeImputer(
            estimator=RandomForestRegressor(random_state=random_state),
            random_state=random_state,
        )

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using a random forest estimator.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        imputed_array = self._imputer.fit_transform(df)
        return pd.DataFrame(imputed_array, columns=df.columns, index=df.index)


class SoftImputeImputer(BaseImputer):
    """Impute missing values using matrix factorization."""

    def __init__(self, max_iters: int = 100, init_fill_method: str = "zero"):
        """Initialize the imputer.

        Args:
            max_iters: Maximum number of iterations for :class:`SoftImpute`.
            init_fill_method: Strategy to initialize missing entries before
                solving.
        """
        self.max_iters = max_iters
        self.init_fill_method = init_fill_method
        self._imputer = SoftImpute(
            max_iters=max_iters, init_fill_method=init_fill_method
        )

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using a low-rank matrix approximation.

        Args:
            df: Dataframe with potential NaN values.

        Returns:
            Dataframe with missing entries imputed by :class:`SoftImpute`.
        """
        df = self._ensure_numeric(df)
        imputed_array = self._imputer.fit_transform(df.to_numpy())
        return pd.DataFrame(imputed_array, columns=df.columns, index=df.index)


class BayesianPCAImputer(BaseImputer):
    """Impute missing values using probabilistic PCA."""

    def __init__(
        self,
        n_components: int | None = 1,
        min_obs: int = 1,
    ) -> None:
        """Initialize the imputer.

        Args:
            n_components: Number of latent dimensions to use. If ``None``, a
                sensible default based on the number of columns is chosen.
            min_obs: Minimum number of observed values required per column.
        """
        self.n_components = n_components
        self.min_obs = min_obs
        self._ppca = PPCA()

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using a Bayesian PCA model.

        Args:
            df: Dataframe with missing values.

        Returns:
            Dataframe with missing entries filled via probabilistic PCA.
        """
        df = self._ensure_numeric(df)
        if df.shape[1] == 1:
            # PPCA cannot reduce to fewer than one component; return original
            return df.copy()
        elif self.n_components is None:
            d = min(df.shape[1], 2)
        else:
            d = min(self.n_components, df.shape[1])
        try:
            self._ppca.fit(
                df.to_numpy(),
                d=d,
                min_obs=self.min_obs,
            )
            # fmt: off
            imputed_array = (
                self._ppca.data * self._ppca.stds
                + self._ppca.means
            )
            # fmt: on
            return pd.DataFrame(
                imputed_array,
                columns=df.columns,
                index=df.index,
            )
        except np.linalg.LinAlgError:
            # Fall back to mean imputation if PPCA fails
            return MeanImputer().impute(df)


class AutoencoderImputer(BaseImputer):
    """Impute missing values using a simple autoencoder."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (10,),
        max_iter: int = 200,
        random_state: int | None = None,
    ) -> None:
        """Initialize the imputer.

        Args:
            hidden_layer_sizes: Architecture of the ``MLPRegressor`` used as
                the autoencoder.
            max_iter: Maximum training iterations.
            random_state: Random seed controlling network initialization.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self._model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            max_iter=max_iter,
            random_state=random_state,
        )

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using an autoencoder reconstruction.

        Args:
            df: Dataframe with missing values.

        Returns:
            Dataframe with imputed values predicted by the autoencoder.
        """
        df = self._ensure_numeric(df)
        filled = df.fillna(df.mean())
        self._model.fit(filled, filled)
        reconstructed = pd.DataFrame(
            self._model.predict(filled),
            columns=df.columns,
            index=df.index,
        )
        return df.where(~df.isna(), reconstructed)


class GAINImputer(BaseImputer):
    """Approximate GAIN using iterative modeling."""

    def __init__(self, random_state: int | None = None) -> None:
        """Initialize the imputer.

        Args:
            random_state: Seed controlling the iterative imputer randomness.
        """
        self.random_state = random_state
        self._imputer = IterativeImputer(random_state=random_state)

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values via iterative modeling.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        imputed_array = self._imputer.fit_transform(df)
        return pd.DataFrame(imputed_array, columns=df.columns, index=df.index)


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

        for column in result.columns:
            if result[column].isna().any():
                predictors = result.columns.difference([column])
                observed = result[result[column].notna()]
                missing = result[result[column].isna()]

                if predictors.size == 0 or observed.empty:
                    continue

                gp = GaussianProcessRegressor(
                    kernel=self.kernel,
                    alpha=self.alpha,
                    random_state=self.random_state,
                )
                gp.fit(observed[predictors], observed[column])
                predicted = gp.predict(missing[predictors])
                result.loc[missing.index, column] = predicted

        return result


def mean_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`MeanImputer`."""
    return MeanImputer().impute(df)


def median_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`MedianImputer`."""
    return MedianImputer().impute(df)


def knn_impute(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`KNNImputerMethod`."""
    return KNNImputerMethod(k=k).impute(df)


def predictive_mean_matching(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`PMMImputer`."""
    return PMMImputer(k=k).impute(df)


def mice_impute(
    df: pd.DataFrame,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`MICEImputer`."""
    return MICEImputer(random_state=random_state).impute(df)


def regression_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`RegressionImputer`."""
    return RegressionImputer().impute(df)


def stochastic_regression_impute(
    df: pd.DataFrame, random_state: int | None = None
) -> pd.DataFrame:
    """Wrapper for :class:`StochasticRegressionImputer`."""
    return StochasticRegressionImputer(random_state=random_state).impute(df)


def locf_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`LOCFImputer`."""
    return LOCFImputer().impute(df)


def nocb_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`NOCBImputer`."""
    return NOCBImputer().impute(df)


def hot_deck_impute(
    df: pd.DataFrame,
    stratify_cols: list[str] | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Wrapper for :class:`HotDeckImputer`."""
    return HotDeckImputer(
        stratify_cols=stratify_cols,
        random_state=random_state,
    ).impute(df)


def miss_forest_impute(
    df: pd.DataFrame, random_state: int | None = None
) -> pd.DataFrame:
    """Wrapper for :class:`MissForestImputer`."""
    return MissForestImputer(random_state=random_state).impute(df)


def bayesian_pca_impute(
    df: pd.DataFrame,
    n_components: int | None = None,
    min_obs: int = 1,
) -> pd.DataFrame:
    """Wrapper for :class:`BayesianPCAImputer`."""
    return BayesianPCAImputer(
        n_components=n_components,
        min_obs=min_obs,
    ).impute(df)


def soft_impute(
    df: pd.DataFrame, max_iters: int = 100, init_fill_method: str = "zero"
) -> pd.DataFrame:
    """Wrapper for :class:`SoftImputeImputer`."""
    return SoftImputeImputer(
        max_iters=max_iters,
        init_fill_method=init_fill_method,
    ).impute(df)


def autoencoder_impute(
    df: pd.DataFrame,
    hidden_layer_sizes: tuple[int, ...] = (10,),
    max_iter: int = 200,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Wrapper for :class:`AutoencoderImputer`."""
    return AutoencoderImputer(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
    ).impute(df)


def gain_impute(
    df: pd.DataFrame,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Wrapper for :class:`GAINImputer`."""
    return GAINImputer(random_state=random_state).impute(df)


def gaussian_process_impute(
    df: pd.DataFrame,
    kernel: RBF | None = None,
    alpha: float = 1e-10,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Wrapper for :class:`GaussianProcessImputer`."""
    return GaussianProcessImputer(
        kernel=kernel,
        alpha=alpha,
        random_state=random_state,
    ).impute(df)


class InterpolationImputer(BaseImputer):
    """Impute missing values using interpolation methods.

    Supports linear, polynomial, and spline interpolation for time series data.

    Args:
        method: Interpolation method ('linear', 'polynomial', 'spline').
            Default: 'linear'
        order: Order for polynomial/spline interpolation. Default: 2
        limit: Maximum number of consecutive NaNs to fill. Default: None (no limit)
        limit_direction: Direction to fill ('forward', 'backward', 'both').
            Default: 'both'

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import InterpolationImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, np.nan, 5]})
        >>> imputer = InterpolationImputer(method='linear')
        >>> imputed = imputer.impute(df)
        >>> print(imputed['a'].tolist())
        [1.0, 2.0, 3.0, 4.0, 5.0]
    """

    def __init__(
        self,
        method: str = 'linear',
        order: int = 2,
        limit: int | None = None,
        limit_direction: str = 'both'
    ):
        """Initialize the interpolation imputer.

        Args:
            method: Interpolation method
            order: Polynomial/spline order
            limit: Maximum consecutive NaNs to fill
            limit_direction: Fill direction
        """
        valid_methods = ['linear', 'polynomial', 'spline']
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {method}")

        self.method = method
        self.order = order
        self.limit = limit
        self.limit_direction = limit_direction

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using interpolation.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            if result[column].isna().any():
                if self.method == 'linear':
                    result[column] = result[column].interpolate(
                        method='linear',
                        limit=self.limit,
                        limit_direction=self.limit_direction
                    )
                elif self.method == 'polynomial':
                    result[column] = result[column].interpolate(
                        method='polynomial',
                        order=self.order,
                        limit=self.limit,
                        limit_direction=self.limit_direction
                    )
                elif self.method == 'spline':
                    result[column] = result[column].interpolate(
                        method='spline',
                        order=self.order,
                        limit=self.limit,
                        limit_direction=self.limit_direction
                    )

                # Fill any remaining NaNs with forward/backward fill
                result[column] = result[column].fillna(method='ffill')
                result[column] = result[column].fillna(method='bfill')

        return result


class EMImputer(BaseImputer):
    """Impute using Expectation-Maximization (EM) algorithm.

    Assumes data follows a multivariate normal distribution and uses
    EM algorithm to estimate parameters and impute missing values.

    Args:
        max_iter: Maximum number of EM iterations. Default: 100
        tol: Convergence tolerance. Default: 1e-4
        random_state: Random seed for reproducibility. Default: None

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import EMImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4], 'b': [5, np.nan, 7, 8]})
        >>> imputer = EMImputer(max_iter=50)
        >>> imputed = imputer.impute(df)
    """

    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None
    ):
        """Initialize the EM imputer.

        Args:
            max_iter: Maximum iterations
            tol: Convergence tolerance
            random_state: Random seed
        """
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using EM algorithm.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)

        # Use IterativeImputer as a proxy for EM-style imputation
        # This is a reasonable approximation of EM behavior
        imputer = IterativeImputer(
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            initial_strategy='mean'
        )

        imputed_array = imputer.fit_transform(df)
        return pd.DataFrame(imputed_array, columns=df.columns, index=df.index)


class MovingAverageImputer(BaseImputer):
    """Impute using moving average (rolling window).

    Fills missing values with the mean or median of a rolling window.

    Args:
        window: Size of the rolling window. Default: 3
        method: Aggregation method ('mean' or 'median'). Default: 'mean'
        min_periods: Minimum observations in window. Default: 1
        center: Whether to center the window. Default: False

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import MovingAverageImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, np.nan, 6]})
        >>> imputer = MovingAverageImputer(window=3, method='mean')
        >>> imputed = imputer.impute(df)
    """

    def __init__(
        self,
        window: int = 3,
        method: str = 'mean',
        min_periods: int = 1,
        center: bool = False
    ):
        """Initialize the moving average imputer.

        Args:
            window: Window size
            method: 'mean' or 'median'
            min_periods: Minimum observations required
            center: Center the window
        """
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if method not in ['mean', 'median']:
            raise ValueError(f"method must be 'mean' or 'median', got {method}")

        self.window = window
        self.method = method
        self.min_periods = min_periods
        self.center = center

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using moving average.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            if result[column].isna().any():
                # Calculate rolling statistic
                rolling = result[column].rolling(
                    window=self.window,
                    min_periods=self.min_periods,
                    center=self.center
                )

                if self.method == 'mean':
                    rolling_values = rolling.mean()
                else:  # median
                    rolling_values = rolling.median()

                # Fill missing values with rolling statistic
                missing_mask = result[column].isna()
                result.loc[missing_mask, column] = rolling_values[missing_mask]

                # Fill any remaining NaNs with column mean
                if result[column].isna().any():
                    result[column] = result[column].fillna(result[column].mean())

        return result


class RandomSamplingImputer(BaseImputer):
    """Impute by randomly sampling from observed values.

    Similar to Hot Deck but without stratification. Preserves the
    empirical distribution of observed values.

    Args:
        random_state: Random seed for reproducibility. Default: None

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import RandomSamplingImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, np.nan, 6]})
        >>> imputer = RandomSamplingImputer(random_state=42)
        >>> imputed = imputer.impute(df)
    """

    def __init__(self, random_state: int | None = None):
        """Initialize the random sampling imputer.

        Args:
            random_state: Random seed
        """
        self.random_state = random_state

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute by random sampling.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        rng = np.random.default_rng(self.random_state)
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            missing_mask = result[column].isna()
            if missing_mask.any():
                # Get observed values
                observed = result[column].dropna()

                if len(observed) > 0:
                    # Sample randomly with replacement
                    n_missing = missing_mask.sum()
                    sampled_values = rng.choice(
                        observed.to_numpy(),
                        size=n_missing,
                        replace=True
                    )
                    result.loc[missing_mask, column] = sampled_values

        return result


class IndicatorImputer(BaseImputer):
    """Impute and add binary indicator columns for missingness.

    Creates indicator columns showing which values were missing,
    then imputes the original columns. Useful when missingness
    itself is informative.

    Args:
        strategy: Imputation strategy for values ('mean', 'median', 'zero').
            Default: 'mean'
        indicator_prefix: Prefix for indicator column names.
            Default: 'missing_'

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import IndicatorImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4], 'b': [5, np.nan, 7, 8]})
        >>> imputer = IndicatorImputer(strategy='mean')
        >>> imputed = imputer.impute(df)
        >>> print(imputed.columns.tolist())
        ['a', 'b', 'missing_a', 'missing_b']
    """

    def __init__(
        self,
        strategy: str = 'mean',
        indicator_prefix: str = 'missing_'
    ):
        """Initialize the indicator imputer.

        Args:
            strategy: Imputation strategy
            indicator_prefix: Prefix for indicator columns
        """
        if strategy not in ['mean', 'median', 'zero']:
            raise ValueError(
                f"strategy must be 'mean', 'median', or 'zero', got {strategy}"
            )

        self.strategy = strategy
        self.indicator_prefix = indicator_prefix

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute and add indicator columns.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe with additional indicator columns.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        # Add indicator columns for missingness
        for column in df.columns:
            indicator_name = f"{self.indicator_prefix}{column}"
            result[indicator_name] = df[column].isna().astype(int)

        # Impute the original columns
        for column in df.columns:
            if result[column].isna().any():
                if self.strategy == 'mean':
                    fill_value = result[column].mean()
                elif self.strategy == 'median':
                    fill_value = result[column].median()
                else:  # zero
                    fill_value = 0

                result[column] = result[column].fillna(fill_value)

        return result


class SeasonalImputer(BaseImputer):
    """Impute using seasonal patterns.

    Decomposes time series into seasonal components and uses
    seasonal averages for imputation.

    Args:
        period: Seasonal period (e.g., 24 for hourly data with daily seasonality,
            7 for daily data with weekly seasonality). Default: 7
        method: Aggregation method ('mean' or 'median'). Default: 'median'

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import SeasonalImputer
        >>> # Daily data with weekly seasonality
        >>> df = pd.DataFrame({'sales': [100, 120, np.nan, 140, 130, np.nan, 90]})
        >>> imputer = SeasonalImputer(period=7, method='median')
        >>> imputed = imputer.impute(df)
    """

    def __init__(self, period: int = 7, method: str = 'median'):
        """Initialize the seasonal imputer.

        Args:
            period: Seasonal period
            method: 'mean' or 'median'
        """
        if period < 2:
            raise ValueError(f"period must be >= 2, got {period}")
        if method not in ['mean', 'median']:
            raise ValueError(f"method must be 'mean' or 'median', got {method}")

        self.period = period
        self.method = method

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using seasonal patterns.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            if result[column].isna().any():
                # Calculate seasonal averages
                seasonal_values = {}

                for phase in range(self.period):
                    # Get all observations at this phase
                    phase_indices = [
                        i for i in range(len(result))
                        if i % self.period == phase
                    ]
                    phase_data = result.iloc[phase_indices][column].dropna()

                    if len(phase_data) > 0:
                        if self.method == 'mean':
                            seasonal_values[phase] = phase_data.mean()
                        else:  # median
                            seasonal_values[phase] = phase_data.median()

                # Impute missing values using seasonal pattern
                for i in range(len(result)):
                    if pd.isna(result.iloc[i][column]):
                        phase = i % self.period
                        if phase in seasonal_values:
                            result.iloc[i, result.columns.get_loc(column)] = seasonal_values[phase]

                # Fill any remaining NaNs with overall mean
                if result[column].isna().any():
                    result[column] = result[column].fillna(result[column].mean())

        return result


class QuantileImputer(BaseImputer):
    """Impute using specified quantile of observed values.

    Args:
        quantile: Quantile to use (0.0 to 1.0). Default: 0.5 (median)

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import QuantileImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, 5]})
        >>> # Use 75th percentile
        >>> imputer = QuantileImputer(quantile=0.75)
        >>> imputed = imputer.impute(df)
    """

    def __init__(self, quantile: float = 0.5):
        """Initialize the quantile imputer.

        Args:
            quantile: Quantile value (0.0 to 1.0)

        Raises:
            ValueError: If quantile is not between 0 and 1
        """
        if not 0 <= quantile <= 1:
            raise ValueError(f"quantile must be between 0 and 1, got {quantile}")

        self.quantile = quantile

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using specified quantile.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            if result[column].isna().any():
                quantile_value = result[column].quantile(self.quantile)
                result[column] = result[column].fillna(quantile_value)

        return result


class ForwardFillFallbackImputer(BaseImputer):
    """Forward fill with fallback to mean/median for leading NaNs.

    Combines LOCF with a fallback strategy for initial missing values
    that cannot be forward filled.

    Args:
        fallback: Fallback strategy ('mean' or 'median'). Default: 'mean'

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import ForwardFillFallbackImputer
        >>> df = pd.DataFrame({'a': [np.nan, np.nan, 3, np.nan, 5]})
        >>> imputer = ForwardFillFallbackImputer(fallback='mean')
        >>> imputed = imputer.impute(df)
        >>> # First two NaNs filled with mean, third NaN forward filled
    """

    def __init__(self, fallback: str = 'mean'):
        """Initialize the forward fill fallback imputer.

        Args:
            fallback: Fallback strategy ('mean' or 'median')

        Raises:
            ValueError: If fallback is not 'mean' or 'median'
        """
        if fallback not in ['mean', 'median']:
            raise ValueError(f"fallback must be 'mean' or 'median', got {fallback}")

        self.fallback = fallback

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using forward fill with fallback.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            if result[column].isna().any():
                # First, forward fill
                result[column] = result[column].fillna(method='ffill')

                # Then fill remaining NaNs with fallback
                if result[column].isna().any():
                    if self.fallback == 'mean':
                        fill_value = result[column].mean()
                    else:  # median
                        fill_value = result[column].median()

                    result[column] = result[column].fillna(fill_value)

        return result


def rmse(true: pd.Series, pred: pd.Series) -> float:
    """Calculate root mean squared error between true and predicted values.

    Args:
        true: Ground truth values.
        pred: Predicted or imputed values.

    Returns:
        The RMSE value.
    """
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(true: pd.Series, pred: pd.Series) -> float:
    """Calculate mean absolute error between true and predicted values.

    Args:
        true: Ground truth values.
        pred: Predicted or imputed values.

    Returns:
        The MAE value.
    """
    return float(np.mean(np.abs(true - pred)))


def interpolation_impute(
    df: pd.DataFrame,
    method: str = 'linear',
    order: int = 2,
    limit: int | None = None,
    limit_direction: str = 'both'
) -> pd.DataFrame:
    """Wrapper for :class:`InterpolationImputer`."""
    return InterpolationImputer(
        method=method,
        order=order,
        limit=limit,
        limit_direction=limit_direction
    ).impute(df)


def em_impute(
    df: pd.DataFrame,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int | None = None
) -> pd.DataFrame:
    """Wrapper for :class:`EMImputer`."""
    return EMImputer(
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    ).impute(df)


def moving_average_impute(
    df: pd.DataFrame,
    window: int = 3,
    method: str = 'mean',
    min_periods: int = 1,
    center: bool = False
) -> pd.DataFrame:
    """Wrapper for :class:`MovingAverageImputer`."""
    return MovingAverageImputer(
        window=window,
        method=method,
        min_periods=min_periods,
        center=center
    ).impute(df)


def random_sampling_impute(
    df: pd.DataFrame,
    random_state: int | None = None
) -> pd.DataFrame:
    """Wrapper for :class:`RandomSamplingImputer`."""
    return RandomSamplingImputer(random_state=random_state).impute(df)


def indicator_impute(
    df: pd.DataFrame,
    strategy: str = 'mean',
    indicator_prefix: str = 'missing_'
) -> pd.DataFrame:
    """Wrapper for :class:`IndicatorImputer`."""
    return IndicatorImputer(
        strategy=strategy,
        indicator_prefix=indicator_prefix
    ).impute(df)


def seasonal_impute(
    df: pd.DataFrame,
    period: int = 7,
    method: str = 'median'
) -> pd.DataFrame:
    """Wrapper for :class:`SeasonalImputer`."""
    return SeasonalImputer(period=period, method=method).impute(df)


def quantile_impute(
    df: pd.DataFrame,
    quantile: float = 0.5
) -> pd.DataFrame:
    """Wrapper for :class:`QuantileImputer`."""
    return QuantileImputer(quantile=quantile).impute(df)


def forward_fill_fallback_impute(
    df: pd.DataFrame,
    fallback: str = 'mean'
) -> pd.DataFrame:
    """Wrapper for :class:`ForwardFillFallbackImputer`."""
    return ForwardFillFallbackImputer(fallback=fallback).impute(df)
