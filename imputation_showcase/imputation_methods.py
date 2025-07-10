"""Imputation methods for the imputation-showcase project."""

from __future__ import annotations

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
    """Impute missing values using column means."""

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        result = df.copy()
        for column in result.columns:
            mean_val = result[column].mean()
            result[column] = result[column].fillna(mean_val)
        return result


class MedianImputer(BaseImputer):
    """Impute missing values using column medians."""

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_numeric(df)
        result = df.copy()
        for column in result.columns:
            median_val = result[column].median()
            result[column] = result[column].fillna(median_val)
        return result


class KNNImputerMethod(BaseImputer):
    """Impute missing values using K-nearest neighbors."""

    def __init__(self, k: int = 5):
        """Initialize the imputer.

        Args:
            k: Number of neighbors to consider.
        """
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
        imputed_array = self._imputer.fit_transform(df)
        return pd.DataFrame(imputed_array, columns=df.columns, index=df.index)


class PMMImputer(BaseImputer):
    """Impute missing values using predictive mean matching (PMM)."""

    def __init__(self, k: int = 5, random_state: int | None = 0):
        """Initialize the imputer.

        Args:
            k: Number of donor candidates to consider.
            random_state: Seed for donor selection randomness.
        """
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

    def __init__(self, random_state: int | None = 0):
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

    def __init__(self, random_state: int | None = 0):
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
        return df.fillna(method="ffill")


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
        return df.fillna(method="bfill")


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
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`MICEImputer`."""
    return MICEImputer(random_state=random_state).impute(df)


def regression_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`RegressionImputer`."""
    return RegressionImputer().impute(df)


def stochastic_regression_impute(
    df: pd.DataFrame, random_state: int | None = 0
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
