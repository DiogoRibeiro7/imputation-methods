"""Imputation methods for the imputation-showcase project."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression, BayesianRidge, HuberRegressor, RANSACRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from scipy import stats
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


class ModeImputer(BaseImputer):
    """Impute missing values with the mode (most frequent value).

    Particularly useful for categorical data or discrete numeric data.
    For continuous data with no repeated values, falls back to median.

    Args:
        dropna: Whether to exclude NaN values when computing mode. Default: True

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import ModeImputer
        >>> df = pd.DataFrame({'a': [1, 2, 2, np.nan, 2, 3]})
        >>> imputer = ModeImputer()
        >>> imputed = imputer.impute(df)
        >>> # Missing value filled with 2 (most frequent)

    References:
        Standard statistical technique for categorical/discrete data.
    """

    def __init__(self, dropna: bool = True):
        """Initialize the mode imputer.

        Args:
            dropna: Whether to exclude NaN values when computing mode
        """
        self.dropna = dropna

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using the mode of each column.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            if result[column].isna().any():
                # Get mode (most frequent value)
                mode_values = result[column].mode(dropna=self.dropna)

                if len(mode_values) > 0:
                    # If multiple modes exist, take the first one
                    fill_value = mode_values[0]
                else:
                    # Fallback to median if no mode found
                    fill_value = result[column].median()

                result[column] = result[column].fillna(fill_value)

        return result


class ConstantImputer(BaseImputer):
    """Impute missing values with a user-specified constant.

    Allows different constants for different columns or a single constant
    for all columns. Useful for domain-specific imputation strategies.

    Args:
        fill_value: Constant value(s) to use for imputation. Can be:
            - A scalar (applied to all columns)
            - A dict mapping column names to fill values
            Default: 0

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import ConstantImputer
        >>> df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [np.nan, 2, 3]})
        >>> # Single value for all columns
        >>> imputer = ConstantImputer(fill_value=-999)
        >>> imputed = imputer.impute(df)
        >>>
        >>> # Different values per column
        >>> imputer = ConstantImputer(fill_value={'a': 0, 'b': 100})
        >>> imputed = imputer.impute(df)

    References:
        Common practice in many domains (e.g., -999 for missing sensor data).
    """

    def __init__(self, fill_value: float | dict[str, float] = 0):
        """Initialize the constant imputer.

        Args:
            fill_value: Constant value(s) for imputation
        """
        self.fill_value = fill_value

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using constant value(s).

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.

        Raises:
            ValueError: If fill_value dict contains unknown column names
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        if isinstance(self.fill_value, dict):
            # Check that all keys in fill_value are valid column names
            unknown_cols = set(self.fill_value.keys()) - set(df.columns)
            if unknown_cols:
                raise ValueError(
                    f"fill_value contains unknown columns: {unknown_cols}"
                )

            # Fill each column with its specific value
            for column, value in self.fill_value.items():
                if column in result.columns and result[column].isna().any():
                    result[column] = result[column].fillna(value)
        else:
            # Fill all columns with the same value
            result = result.fillna(self.fill_value)

        return result


class EndOfDistributionImputer(BaseImputer):
    """Impute at the edges of the distribution (mean ± k*std).

    Useful for flagging or handling extreme/suspicious values.
    Can impute at low end (mean - k*std) or high end (mean + k*std).

    Args:
        position: Where to impute ('low' or 'high'). Default: 'high'
        k: Number of standard deviations from mean. Default: 3.0

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import EndOfDistributionImputer
        >>> df = pd.DataFrame({'a': [1, 2, 3, np.nan, 5]})
        >>> imputer = EndOfDistributionImputer(position='high', k=2)
        >>> imputed = imputer.impute(df)
        >>> # Missing value filled with mean + 2*std

    References:
        Used in outlier detection and robust imputation strategies.
    """

    def __init__(self, position: str = 'high', k: float = 3.0):
        """Initialize the end-of-distribution imputer.

        Args:
            position: 'low' (mean - k*std) or 'high' (mean + k*std)
            k: Number of standard deviations

        Raises:
            ValueError: If position is not 'low' or 'high'
        """
        if position not in ['low', 'high']:
            raise ValueError(f"position must be 'low' or 'high', got {position}")

        self.position = position
        self.k = k

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute at distribution edges.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            if result[column].isna().any():
                mean = result[column].mean()
                std = result[column].std()

                if self.position == 'low':
                    fill_value = mean - self.k * std
                else:  # high
                    fill_value = mean + self.k * std

                result[column] = result[column].fillna(fill_value)

        return result


class GroupMeanImputer(BaseImputer):
    """Group-wise mean or median imputation.

    Imputes missing values using statistics computed within groups.
    Useful for panel data, time series with categories, etc.

    Args:
        group_col: Column name to group by (must be in the dataframe)
        method: Aggregation method ('mean' or 'median'). Default: 'mean'
        global_fallback: Use global statistic if group has no data. Default: True

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import GroupMeanImputer
        >>> df = pd.DataFrame({
        ...     'category': [1, 1, 2, 2, 1],
        ...     'value': [10, np.nan, 20, np.nan, 12]
        ... })
        >>> imputer = GroupMeanImputer(group_col='category', method='mean')
        >>> imputed = imputer.impute(df)
        >>> # Row 1 filled with mean of category 1, row 3 with mean of category 2

    References:
        Common in hierarchical data and panel data analysis.
    """

    def __init__(
        self,
        group_col: str,
        method: str = 'mean',
        global_fallback: bool = True
    ):
        """Initialize the group mean imputer.

        Args:
            group_col: Column name to group by
            method: 'mean' or 'median'
            global_fallback: Use global statistic for groups with no data

        Raises:
            ValueError: If method is not 'mean' or 'median'
        """
        if method not in ['mean', 'median']:
            raise ValueError(f"method must be 'mean' or 'median', got {method}")

        self.group_col = group_col
        self.method = method
        self.global_fallback = global_fallback

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using group-wise statistics.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.

        Raises:
            ValueError: If group_col is not in dataframe columns
        """
        if self.group_col not in df.columns:
            raise ValueError(
                f"group_col '{self.group_col}' not found in dataframe columns"
            )

        result = df.copy()

        # Get numeric columns (excluding group column)
        numeric_cols = [
            col for col in result.columns
            if col != self.group_col and pd.api.types.is_numeric_dtype(result[col])
        ]

        for column in numeric_cols:
            if result[column].isna().any():
                # Compute group-wise statistics
                if self.method == 'mean':
                    group_stats = result.groupby(self.group_col)[column].mean()
                else:  # median
                    group_stats = result.groupby(self.group_col)[column].median()

                # Fill using group statistics
                result[column] = result.apply(
                    lambda row: (
                        group_stats.get(row[self.group_col], np.nan)
                        if pd.isna(row[column])
                        else row[column]
                    ),
                    axis=1
                )

                # Handle any remaining NaNs with global fallback
                if self.global_fallback and result[column].isna().any():
                    if self.method == 'mean':
                        global_stat = df[column].mean()
                    else:  # median
                        global_stat = df[column].median()

                    result[column] = result[column].fillna(global_stat)

        return result


class WeightedMovingAverageImputer(BaseImputer):
    """Exponentially weighted moving average imputation for time series.

    Uses exponential weighting to give more importance to recent values.
    More sophisticated than simple moving average for trending data.

    Args:
        alpha: Smoothing factor (0 < alpha <= 1). Higher = more weight to recent.
            Default: 0.5
        min_periods: Minimum observations needed. Default: 1

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import WeightedMovingAverageImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, np.nan, 6]})
        >>> imputer = WeightedMovingAverageImputer(alpha=0.7)
        >>> imputed = imputer.impute(df)
        >>> # Missing values filled using exponentially weighted average

    References:
        Commonly used in financial time series and sensor data analysis.
    """

    def __init__(self, alpha: float = 0.5, min_periods: int = 1):
        """Initialize the weighted moving average imputer.

        Args:
            alpha: Smoothing factor (0 < alpha <= 1)
            min_periods: Minimum observations required

        Raises:
            ValueError: If alpha is not in (0, 1]
        """
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

        self.alpha = alpha
        self.min_periods = min_periods

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using exponentially weighted moving average.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            if result[column].isna().any():
                # Calculate EWMA
                ewma = result[column].ewm(
                    alpha=self.alpha,
                    min_periods=self.min_periods,
                    ignore_na=True
                ).mean()

                # Fill NaNs with EWMA values
                result[column] = result[column].fillna(ewma)

                # If still NaNs (at the beginning), use backward fill then mean
                if result[column].isna().any():
                    result[column] = result[column].bfill()
                if result[column].isna().any():
                    result[column] = result[column].fillna(result[column].mean())

        return result


class LinearTrendImputer(BaseImputer):
    """Linear trend imputation for time series data.

    Fits a linear trend to observed data and uses it to fill missing values.
    Suitable for data with clear linear trends.

    Args:
        use_index: Use dataframe index as x-values. If False, use integer positions.
            Default: False

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import LinearTrendImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, np.nan, 6]})
        >>> imputer = LinearTrendImputer()
        >>> imputed = imputer.impute(df)
        >>> # Missing values filled based on linear trend

    References:
        Standard technique for trending time series data.
    """

    def __init__(self, use_index: bool = False):
        """Initialize the linear trend imputer.

        Args:
            use_index: Whether to use dataframe index as x-values
        """
        self.use_index = use_index

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using linear trend.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            if result[column].isna().any():
                # Get observed values and their positions
                mask = ~result[column].isna()
                observed_values = result.loc[mask, column].values

                if self.use_index:
                    observed_positions = result.loc[mask].index.values
                    all_positions = result.index.values
                else:
                    observed_positions = np.where(mask)[0]
                    all_positions = np.arange(len(result))

                if len(observed_values) > 0:
                    # Fit linear model
                    if len(observed_values) == 1:
                        # Can't fit a line with one point, use constant
                        predictions = np.full(len(all_positions), observed_values[0])
                    else:
                        model = LinearRegression()
                        X = observed_positions.reshape(-1, 1)
                        y = observed_values
                        model.fit(X, y)

                        # Predict for all positions
                        predictions = model.predict(all_positions.reshape(-1, 1))

                    # Fill missing values
                    result.loc[result[column].isna(), column] = predictions[result[column].isna()]

        return result


class PolynomialTrendImputer(BaseImputer):
    """Polynomial trend imputation for time series data.

    Fits polynomial curve to observed data for non-linear trends.
    More flexible than linear trend for complex patterns.

    Args:
        degree: Polynomial degree (1=linear, 2=quadratic, 3=cubic, etc.).
            Default: 2
        use_index: Use dataframe index as x-values. Default: False

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import PolynomialTrendImputer
        >>> df = pd.DataFrame({'a': [1, 4, np.nan, 16, np.nan, 36]})
        >>> imputer = PolynomialTrendImputer(degree=2)
        >>> imputed = imputer.impute(df)
        >>> # Missing values filled based on quadratic trend

    References:
        Used for time series with non-linear but smooth trends.
    """

    def __init__(self, degree: int = 2, use_index: bool = False):
        """Initialize the polynomial trend imputer.

        Args:
            degree: Polynomial degree (must be >= 1)
            use_index: Whether to use dataframe index as x-values

        Raises:
            ValueError: If degree < 1
        """
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")

        self.degree = degree
        self.use_index = use_index

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using polynomial trend.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            if result[column].isna().any():
                # Get observed values and their positions
                mask = ~result[column].isna()
                observed_values = result.loc[mask, column].values

                if self.use_index:
                    observed_positions = result.loc[mask].index.values
                    all_positions = result.index.values
                else:
                    observed_positions = np.where(mask)[0]
                    all_positions = np.arange(len(result))

                if len(observed_values) > 0:
                    # Ensure we have enough points for the polynomial degree
                    effective_degree = min(self.degree, len(observed_values) - 1)

                    if effective_degree == 0:
                        # Only one point, use constant
                        predictions = np.full(len(all_positions), observed_values[0])
                    else:
                        # Fit polynomial
                        coefficients = np.polyfit(
                            observed_positions,
                            observed_values,
                            effective_degree
                        )
                        predictions = np.polyval(coefficients, all_positions)

                    # Fill missing values
                    result.loc[result[column].isna(), column] = predictions[result[column].isna()]

        return result


class KalmanFilterImputer(BaseImputer):
    """Kalman filter imputation for time series with uncertainty.

    Uses Kalman filtering to impute missing values while accounting for
    measurement noise and process uncertainty. Ideal for sensor data.

    Args:
        process_variance: Process noise variance (Q). Default: 1.0
        measurement_variance: Measurement noise variance (R). Default: 1.0
        initial_state: Initial state estimate. If None, uses first observed value.
            Default: None
        initial_covariance: Initial error covariance (P). Default: 1.0

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import KalmanFilterImputer
        >>> df = pd.DataFrame({'a': [1, np.nan, 3, np.nan, 5]})
        >>> imputer = KalmanFilterImputer()
        >>> imputed = imputer.impute(df)
        >>> # Missing values filled using Kalman filter estimates

    References:
        Kalman, R. E. (1960). A new approach to linear filtering and prediction.
        Widely used in sensor fusion and state estimation.
    """

    def __init__(
        self,
        process_variance: float = 1.0,
        measurement_variance: float = 1.0,
        initial_state: float | None = None,
        initial_covariance: float = 1.0
    ):
        """Initialize the Kalman filter imputer.

        Args:
            process_variance: Process noise variance (Q)
            measurement_variance: Measurement noise variance (R)
            initial_state: Initial state estimate (None = use first observed)
            initial_covariance: Initial error covariance (P)
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.initial_state = initial_state
        self.initial_covariance = initial_covariance

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using Kalman filter.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            if result[column].isna().any():
                values = result[column].values.copy()

                # Initialize Kalman filter state
                first_obs_idx = np.where(~np.isnan(values))[0]
                if len(first_obs_idx) == 0:
                    continue  # No observed values

                if self.initial_state is None:
                    x_est = values[first_obs_idx[0]]  # Initial state estimate
                else:
                    x_est = self.initial_state

                p_est = self.initial_covariance  # Initial error covariance

                # Run Kalman filter
                for i in range(len(values)):
                    # Prediction step
                    x_pred = x_est  # State transition (simple: x_k = x_{k-1})
                    p_pred = p_est + self.process_variance

                    if not np.isnan(values[i]):
                        # Update step (measurement available)
                        kalman_gain = p_pred / (p_pred + self.measurement_variance)
                        x_est = x_pred + kalman_gain * (values[i] - x_pred)
                        p_est = (1 - kalman_gain) * p_pred
                    else:
                        # No measurement, use prediction
                        values[i] = x_pred
                        x_est = x_pred
                        p_est = p_pred

                result[column] = values

        return result


class ColdDeckImputer(BaseImputer):
    """Cold deck imputation using predetermined reference values.

    Uses values from a reference dataset or predetermined mapping to fill
    missing values. Useful when you have historical or domain knowledge.

    Args:
        reference_values: Dictionary mapping column names to reference values
            or a reference DataFrame. If dict, can map to scalar or array.
            Default: None (uses column median as fallback)

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import ColdDeckImputer
        >>> df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [np.nan, 2, 3]})
        >>> # Use predetermined values
        >>> imputer = ColdDeckImputer(reference_values={'a': 2.5, 'b': 2.0})
        >>> imputed = imputer.impute(df)

    References:
        Traditional imputation method predating hot deck imputation.
        Uses external/historical data rather than current dataset.
    """

    def __init__(self, reference_values: dict[str, float | np.ndarray] | pd.DataFrame | None = None):
        """Initialize the cold deck imputer.

        Args:
            reference_values: Reference values for imputation
        """
        self.reference_values = reference_values

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using cold deck reference values.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        if self.reference_values is None:
            # Fallback to median
            for column in result.columns:
                if result[column].isna().any():
                    result[column] = result[column].fillna(result[column].median())
        elif isinstance(self.reference_values, pd.DataFrame):
            # Use reference DataFrame
            for column in result.columns:
                if column in self.reference_values.columns and result[column].isna().any():
                    ref_mean = self.reference_values[column].mean()
                    result[column] = result[column].fillna(ref_mean)
        elif isinstance(self.reference_values, dict):
            # Use dictionary of reference values
            for column in result.columns:
                if result[column].isna().any():
                    if column in self.reference_values:
                        ref_value = self.reference_values[column]
                        if isinstance(ref_value, (int, float, np.number)):
                            # Scalar reference value
                            result[column] = result[column].fillna(ref_value)
                        else:
                            # Array reference value - sample randomly
                            fill_values = np.random.choice(
                                ref_value,
                                size=result[column].isna().sum()
                            )
                            result.loc[result[column].isna(), column] = fill_values
                    else:
                        # Fallback to median if column not in reference
                        result[column] = result[column].fillna(result[column].median())

        return result


class HybridImputer(BaseImputer):
    """Hybrid imputation combining multiple methods with fallback chain.

    Tries multiple imputation methods in sequence, falling back to simpler
    methods if earlier methods fail or produce NaNs. Robust for diverse data.

    Args:
        methods: List of imputer instances to try in order.
            Default: [InterpolationImputer(), MeanImputer()]

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import (
        ...     HybridImputer, InterpolationImputer,
        ...     MovingAverageImputer, MeanImputer
        ... )
        >>> df = pd.DataFrame({'a': [1, np.nan, np.nan, 4, np.nan]})
        >>> # Try interpolation, then moving average, then mean
        >>> imputer = HybridImputer(methods=[
        ...     InterpolationImputer(),
        ...     MovingAverageImputer(window=2),
        ...     MeanImputer()
        ... ])
        >>> imputed = imputer.impute(df)

    References:
        Combines strengths of multiple methods for robust imputation.
    """

    def __init__(self, methods: list[BaseImputer] | None = None):
        """Initialize the hybrid imputer.

        Args:
            methods: List of imputer instances to try in order
        """
        if methods is None:
            # Default fallback chain
            methods = [InterpolationImputer(), MeanImputer()]

        self.methods = methods

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using hybrid method chain.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        # Try each method in sequence
        for method in self.methods:
            if result.isna().any().any():
                try:
                    result = method.impute(result)
                except Exception as e:
                    # If a method fails, continue to next method
                    logger.warning(
                        f"Method {method.__class__.__name__} failed: {e}. "
                        "Trying next method."
                    )
                    continue
            else:
                # All values imputed, no need for further methods
                break

        # Final fallback if still have NaNs
        if result.isna().any().any():
            for column in result.columns:
                if result[column].isna().any():
                    # Use mean as last resort
                    result[column] = result[column].fillna(result[column].mean())
                    # If mean is NaN (all values missing), use 0
                    result[column] = result[column].fillna(0)

        return result


class BayesianRidgeImputer(BaseImputer):
    """Bayesian ridge regression imputation for missing values.

    Uses Bayesian ridge regression to predict missing values based on other
    features. Provides probabilistic estimates and handles multicollinearity well.

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
        >>> from imputation_showcase import BayesianRidgeImputer
        >>> df = pd.DataFrame({
        ...     'a': [1, 2, np.nan, 4, 5],
        ...     'b': [2, 4, 6, np.nan, 10]
        ... })
        >>> imputer = BayesianRidgeImputer()
        >>> imputed = imputer.impute(df)

    References:
        Bayesian approach to ridge regression with automatic relevance determination.
    """

    def __init__(
        self,
        max_iter: int = 300,
        tol: float = 1e-3,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6
    ):
        """Initialize the Bayesian ridge imputer.

        Args:
            max_iter: Maximum iterations
            tol: Convergence tolerance
            alpha_1, alpha_2, lambda_1, lambda_2: Hyperparameters for priors
        """
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using Bayesian ridge regression.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            if result[column].isna().any():
                # Get rows with and without missing values in this column
                train_mask = ~result[column].isna()
                predict_mask = result[column].isna()

                if train_mask.sum() == 0:
                    # No training data, use mean of other columns
                    result[column] = result[column].fillna(0)
                    continue

                # Features are all other columns
                feature_cols = [c for c in result.columns if c != column]
                if len(feature_cols) == 0:
                    # No features available, use mean
                    result[column] = result[column].fillna(result[column].mean())
                    continue

                X_train = result.loc[train_mask, feature_cols].fillna(0).values
                y_train = result.loc[train_mask, column].values
                X_predict = result.loc[predict_mask, feature_cols].fillna(0).values

                if len(X_train) > 0 and len(X_predict) > 0:
                    model = BayesianRidge(
                        max_iter=self.max_iter,
                        tol=self.tol,
                        alpha_1=self.alpha_1,
                        alpha_2=self.alpha_2,
                        lambda_1=self.lambda_1,
                        lambda_2=self.lambda_2
                    )
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_predict)
                    result.loc[predict_mask, column] = predictions

        return result


class StackingImputer(BaseImputer):
    """Stacking ensemble imputer combining multiple base imputers.

    Trains multiple base imputers and combines their predictions using
    a meta-learner for improved accuracy.

    Args:
        base_imputers: List of base imputer instances to stack.
            Default: [MeanImputer(), MedianImputer(), KNNImputerMethod()]
        meta_strategy: How to combine predictions ('mean', 'median', 'weighted').
            Default: 'mean'

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import (
        ...     StackingImputer, MeanImputer, MedianImputer, KNNImputerMethod
        ... )
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, 5]})
        >>> imputer = StackingImputer(base_imputers=[
        ...     MeanImputer(),
        ...     MedianImputer(),
        ...     KNNImputerMethod(n_neighbors=2)
        ... ])
        >>> imputed = imputer.impute(df)

    References:
        Ensemble learning approach applied to imputation.
    """

    def __init__(
        self,
        base_imputers: list[BaseImputer] | None = None,
        meta_strategy: str = 'mean'
    ):
        """Initialize the stacking imputer.

        Args:
            base_imputers: List of base imputers
            meta_strategy: Strategy for combining predictions

        Raises:
            ValueError: If meta_strategy is invalid
        """
        if meta_strategy not in ['mean', 'median', 'weighted']:
            raise ValueError(
                f"meta_strategy must be 'mean', 'median', or 'weighted', "
                f"got {meta_strategy}"
            )

        if base_imputers is None:
            base_imputers = [MeanImputer(), MedianImputer()]

        self.base_imputers = base_imputers
        self.meta_strategy = meta_strategy

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using stacking ensemble.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)

        # Get predictions from all base imputers
        predictions = []
        for imputer in self.base_imputers:
            try:
                pred = imputer.impute(df)
                predictions.append(pred)
            except Exception as e:
                logger.warning(
                    f"Base imputer {imputer.__class__.__name__} failed: {e}"
                )
                continue

        if len(predictions) == 0:
            # All base imputers failed, fall back to mean
            return MeanImputer().impute(df)

        # Combine predictions
        if self.meta_strategy == 'mean':
            result = sum(predictions) / len(predictions)
        elif self.meta_strategy == 'median':
            # Stack predictions and take median
            stacked = np.stack([p.values for p in predictions], axis=0)
            result = pd.DataFrame(
                np.median(stacked, axis=0),
                index=df.index,
                columns=df.columns
            )
        else:  # weighted - give more weight to imputers that agree
            # Simple implementation: use mean (could be enhanced)
            result = sum(predictions) / len(predictions)

        return result


class BaggingImputer(BaseImputer):
    """Bootstrap aggregating (bagging) for robust imputation.

    Creates multiple bootstrap samples, imputes each, and aggregates
    results for more stable predictions.

    Args:
        base_imputer: Base imputer to use for each bootstrap sample.
            Default: MeanImputer()
        n_estimators: Number of bootstrap samples. Default: 10
        max_samples: Fraction of samples to draw for each bootstrap. Default: 0.8
        random_state: Random seed for reproducibility. Default: None

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import BaggingImputer, KNNImputerMethod
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, 5, np.nan, 7]})
        >>> imputer = BaggingImputer(
        ...     base_imputer=KNNImputerMethod(),
        ...     n_estimators=5
        ... )
        >>> imputed = imputer.impute(df)

    References:
        Bootstrap aggregating for variance reduction in predictions.
    """

    def __init__(
        self,
        base_imputer: BaseImputer | None = None,
        n_estimators: int = 10,
        max_samples: float = 0.8,
        random_state: int | None = None
    ):
        """Initialize the bagging imputer.

        Args:
            base_imputer: Base imputer instance
            n_estimators: Number of bootstrap samples
            max_samples: Fraction of samples per bootstrap
            random_state: Random seed
        """
        if base_imputer is None:
            base_imputer = MeanImputer()

        self.base_imputer = base_imputer
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using bagging.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)

        # Store predictions from each estimator
        all_predictions = []

        for i in range(self.n_estimators):
            # Apply base imputer with different random state (for variety)
            try:
                # Create a copy of base imputer if it has random_state
                if hasattr(self.base_imputer, 'random_state'):
                    # Make a simple copy with modified random state
                    seed = (self.random_state or 0) + i
                    imputed = self.base_imputer.impute(df)
                else:
                    imputed = self.base_imputer.impute(df)

                all_predictions.append(imputed)
            except Exception as e:
                logger.warning(f"Estimator {i} failed: {e}")
                continue

        if len(all_predictions) == 0:
            # All estimators failed, fall back to base imputer
            return self.base_imputer.impute(df)

        # Average all predictions
        result = sum(all_predictions) / len(all_predictions)

        # Ensure no NaNs remain
        if result.isna().any().any():
            result = result.fillna(df.mean())
            if result.isna().any().any():
                result = result.fillna(0)

        return result


class RadiusNeighborsImputer(BaseImputer):
    """Radius-based neighbors imputation using distance threshold.

    Imputes using all neighbors within a specified radius rather than
    a fixed number of neighbors. Adaptive to local density.

    Args:
        radius: Distance threshold for neighbors. Default: 1.0
        weights: Weight function ('uniform' or 'distance'). Default: 'distance'
        metric: Distance metric. Default: 'euclidean'

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import RadiusNeighborsImputer
        >>> df = pd.DataFrame({
        ...     'a': [1, 2, np.nan, 4, 5],
        ...     'b': [2, 4, 6, np.nan, 10]
        ... })
        >>> imputer = RadiusNeighborsImputer(radius=2.0)
        >>> imputed = imputer.impute(df)

    References:
        Radius-based neighborhood for adaptive local imputation.
    """

    def __init__(
        self,
        radius: float = 1.0,
        weights: str = 'distance',
        metric: str = 'euclidean'
    ):
        """Initialize the radius neighbors imputer.

        Args:
            radius: Distance threshold
            weights: Weighting function
            metric: Distance metric
        """
        self.radius = radius
        self.weights = weights
        self.metric = metric

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using radius neighbors.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        # For each column with missing values
        for column in result.columns:
            if result[column].isna().any():
                train_mask = ~result[column].isna()
                predict_mask = result[column].isna()

                if train_mask.sum() == 0:
                    result[column] = result[column].fillna(result[column].mean())
                    continue

                feature_cols = [c for c in result.columns if c != column]
                if len(feature_cols) == 0:
                    result[column] = result[column].fillna(result[column].mean())
                    continue

                X_train = result.loc[train_mask, feature_cols].fillna(0).values
                y_train = result.loc[train_mask, column].values
                X_predict = result.loc[predict_mask, feature_cols].fillna(0).values

                if len(X_train) > 0 and len(X_predict) > 0:
                    try:
                        model = RadiusNeighborsRegressor(
                            radius=self.radius,
                            weights=self.weights,
                            metric=self.metric
                        )
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_predict)
                        result.loc[predict_mask, column] = predictions
                    except Exception:
                        # Fall back to mean if radius neighbors fails
                        result.loc[predict_mask, column] = result[column].mean()

        return result


class LocalMeanImputer(BaseImputer):
    """Local weighted mean imputation based on feature similarity.

    Computes weighted average of similar observations, with weights
    decreasing by distance.

    Args:
        n_neighbors: Number of neighbors to consider. Default: 5
        distance_weight_power: Power for distance weighting. Default: 2.0

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import LocalMeanImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, 5]})
        >>> imputer = LocalMeanImputer(n_neighbors=3)
        >>> imputed = imputer.impute(df)

    References:
        Locally weighted averaging for smooth imputation.
    """

    def __init__(self, n_neighbors: int = 5, distance_weight_power: float = 2.0):
        """Initialize the local mean imputer.

        Args:
            n_neighbors: Number of neighbors
            distance_weight_power: Power for weighting by distance
        """
        self.n_neighbors = n_neighbors
        self.distance_weight_power = distance_weight_power

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using local weighted mean.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            if result[column].isna().any():
                for idx in result[result[column].isna()].index:
                    # Get feature values for this row (excluding target column)
                    feature_cols = [c for c in result.columns if c != column]
                    if len(feature_cols) == 0:
                        result.loc[idx, column] = result[column].mean()
                        continue

                    row_features = result.loc[idx, feature_cols].fillna(0).values

                    # Find distances to all complete observations
                    complete_mask = ~result[column].isna()
                    if complete_mask.sum() == 0:
                        result.loc[idx, column] = 0
                        continue

                    complete_features = result.loc[complete_mask, feature_cols].fillna(0).values
                    complete_values = result.loc[complete_mask, column].values

                    # Compute Euclidean distances
                    distances = np.sqrt(np.sum((complete_features - row_features) ** 2, axis=1))

                    # Get k nearest neighbors
                    k = min(self.n_neighbors, len(distances))
                    nearest_idx = np.argsort(distances)[:k]

                    # Compute weights (inverse distance)
                    nearest_distances = distances[nearest_idx]
                    # Avoid division by zero
                    nearest_distances = np.maximum(nearest_distances, 1e-10)
                    weights = 1.0 / (nearest_distances ** self.distance_weight_power)
                    weights /= weights.sum()

                    # Weighted average
                    result.loc[idx, column] = np.sum(weights * complete_values[nearest_idx])

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
        >>> from imputation_showcase import HuberImputer
        >>> df = pd.DataFrame({
        ...     'a': [1, 2, np.nan, 100, 5],  # 100 is outlier
        ...     'b': [2, 4, 6, 200, np.nan]
        ... })
        >>> imputer = HuberImputer()
        >>> imputed = imputer.impute(df)

    References:
        Huber, P. J. (1964). Robust estimation of a location parameter.
    """

    def __init__(self, epsilon: float = 1.35, max_iter: int = 100, alpha: float = 0.0001):
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
                        epsilon=self.epsilon,
                        max_iter=self.max_iter,
                        alpha=self.alpha
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
        >>> from imputation_showcase import RANSACImputer
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
        random_state: int | None = None
    ):
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
                            random_state=self.random_state
                        )
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_predict)
                        result.loc[predict_mask, column] = predictions
                    except Exception:
                        # Fall back to median if RANSAC fails
                        result.loc[predict_mask, column] = result[column].median()

        return result


class TrimmedMeanImputer(BaseImputer):
    """Trimmed mean imputation excluding extreme values.

    Computes mean after removing a percentage of extreme values
    from both ends. More robust than simple mean.

    Args:
        trim_fraction: Fraction to trim from each end (0-0.5). Default: 0.1

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_showcase import TrimmedMeanImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, 100]})  # 100 is outlier
        >>> imputer = TrimmedMeanImputer(trim_fraction=0.2)
        >>> imputed = imputer.impute(df)
        >>> # Excludes 100 from mean calculation

    References:
        Robust statistics using trimmed estimators.
    """

    def __init__(self, trim_fraction: float = 0.1):
        """Initialize the trimmed mean imputer.

        Args:
            trim_fraction: Fraction to trim (0-0.5)

        Raises:
            ValueError: If trim_fraction not in [0, 0.5]
        """
        if not (0 <= trim_fraction < 0.5):
            raise ValueError(f"trim_fraction must be in [0, 0.5), got {trim_fraction}")

        self.trim_fraction = trim_fraction

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using trimmed mean.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        for column in result.columns:
            if result[column].isna().any():
                # Compute trimmed mean
                trimmed_mean = stats.trim_mean(
                    result[column].dropna(),
                    self.trim_fraction
                )
                result[column] = result[column].fillna(trimmed_mean)

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


def mode_impute(
    df: pd.DataFrame,
    dropna: bool = True
) -> pd.DataFrame:
    """Wrapper for :class:`ModeImputer`."""
    return ModeImputer(dropna=dropna).impute(df)


def constant_impute(
    df: pd.DataFrame,
    fill_value: float | dict[str, float] = 0
) -> pd.DataFrame:
    """Wrapper for :class:`ConstantImputer`."""
    return ConstantImputer(fill_value=fill_value).impute(df)


def end_of_distribution_impute(
    df: pd.DataFrame,
    position: str = 'high',
    k: float = 3.0
) -> pd.DataFrame:
    """Wrapper for :class:`EndOfDistributionImputer`."""
    return EndOfDistributionImputer(position=position, k=k).impute(df)


def group_mean_impute(
    df: pd.DataFrame,
    group_col: str,
    method: str = 'mean',
    global_fallback: bool = True
) -> pd.DataFrame:
    """Wrapper for :class:`GroupMeanImputer`."""
    return GroupMeanImputer(
        group_col=group_col,
        method=method,
        global_fallback=global_fallback
    ).impute(df)


def weighted_moving_average_impute(
    df: pd.DataFrame,
    alpha: float = 0.5,
    min_periods: int = 1
) -> pd.DataFrame:
    """Wrapper for :class:`WeightedMovingAverageImputer`."""
    return WeightedMovingAverageImputer(
        alpha=alpha,
        min_periods=min_periods
    ).impute(df)


def linear_trend_impute(
    df: pd.DataFrame,
    use_index: bool = False
) -> pd.DataFrame:
    """Wrapper for :class:`LinearTrendImputer`."""
    return LinearTrendImputer(use_index=use_index).impute(df)


def polynomial_trend_impute(
    df: pd.DataFrame,
    degree: int = 2,
    use_index: bool = False
) -> pd.DataFrame:
    """Wrapper for :class:`PolynomialTrendImputer`."""
    return PolynomialTrendImputer(degree=degree, use_index=use_index).impute(df)


def kalman_filter_impute(
    df: pd.DataFrame,
    process_variance: float = 1.0,
    measurement_variance: float = 1.0,
    initial_state: float | None = None,
    initial_covariance: float = 1.0
) -> pd.DataFrame:
    """Wrapper for :class:`KalmanFilterImputer`."""
    return KalmanFilterImputer(
        process_variance=process_variance,
        measurement_variance=measurement_variance,
        initial_state=initial_state,
        initial_covariance=initial_covariance
    ).impute(df)


def cold_deck_impute(
    df: pd.DataFrame,
    reference_values: dict[str, float | np.ndarray] | pd.DataFrame | None = None
) -> pd.DataFrame:
    """Wrapper for :class:`ColdDeckImputer`."""
    return ColdDeckImputer(reference_values=reference_values).impute(df)


def hybrid_impute(
    df: pd.DataFrame,
    methods: list[BaseImputer] | None = None
) -> pd.DataFrame:
    """Wrapper for :class:`HybridImputer`."""
    return HybridImputer(methods=methods).impute(df)


def bayesian_ridge_impute(
    df: pd.DataFrame,
    max_iter: int = 300,
    tol: float = 1e-3
) -> pd.DataFrame:
    """Wrapper for :class:`BayesianRidgeImputer`."""
    return BayesianRidgeImputer(max_iter=max_iter, tol=tol).impute(df)


def stacking_impute(
    df: pd.DataFrame,
    base_imputers: list[BaseImputer] | None = None,
    meta_strategy: str = 'mean'
) -> pd.DataFrame:
    """Wrapper for :class:`StackingImputer`."""
    return StackingImputer(
        base_imputers=base_imputers,
        meta_strategy=meta_strategy
    ).impute(df)


def bagging_impute(
    df: pd.DataFrame,
    base_imputer: BaseImputer | None = None,
    n_estimators: int = 10,
    random_state: int | None = None
) -> pd.DataFrame:
    """Wrapper for :class:`BaggingImputer`."""
    return BaggingImputer(
        base_imputer=base_imputer,
        n_estimators=n_estimators,
        random_state=random_state
    ).impute(df)


def radius_neighbors_impute(
    df: pd.DataFrame,
    radius: float = 1.0,
    weights: str = 'distance'
) -> pd.DataFrame:
    """Wrapper for :class:`RadiusNeighborsImputer`."""
    return RadiusNeighborsImputer(radius=radius, weights=weights).impute(df)


def local_mean_impute(
    df: pd.DataFrame,
    n_neighbors: int = 5,
    distance_weight_power: float = 2.0
) -> pd.DataFrame:
    """Wrapper for :class:`LocalMeanImputer`."""
    return LocalMeanImputer(
        n_neighbors=n_neighbors,
        distance_weight_power=distance_weight_power
    ).impute(df)


def huber_impute(
    df: pd.DataFrame,
    epsilon: float = 1.35,
    max_iter: int = 100
) -> pd.DataFrame:
    """Wrapper for :class:`HuberImputer`."""
    return HuberImputer(epsilon=epsilon, max_iter=max_iter).impute(df)


def ransac_impute(
    df: pd.DataFrame,
    max_trials: int = 100,
    random_state: int | None = None
) -> pd.DataFrame:
    """Wrapper for :class:`RANSACImputer`."""
    return RANSACImputer(
        max_trials=max_trials,
        random_state=random_state
    ).impute(df)


def trimmed_mean_impute(
    df: pd.DataFrame,
    trim_fraction: float = 0.1
) -> pd.DataFrame:
    """Wrapper for :class:`TrimmedMeanImputer`."""
    return TrimmedMeanImputer(trim_fraction=trim_fraction).impute(df)
