"""Imputation methods for the imputation-showcase project."""

from __future__ import annotations

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


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
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes):
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
                    imputed_val = donors[column].sample(1, random_state=self.random_state).iloc[0]
                    result.at[i, column] = imputed_val

        return result


class MICEImputer(BaseImputer):
    """Impute missing data using Multiple Imputation by Chained Equations (MICE)."""

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
            Imputed dataframe with missing values filled by regression predictions.
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
                residuals = observed[column] - reg.predict(observed[predictors])
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


def mice_impute(df: pd.DataFrame, random_state: int | None = 0) -> pd.DataFrame:
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

