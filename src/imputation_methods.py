"""Imputation methods for the imputation-showcase project."""

from __future__ import annotations

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression


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


def mean_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`MeanImputer`."""
    return MeanImputer().impute(df)


def knn_impute(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`KNNImputerMethod`."""
    return KNNImputerMethod(k=k).impute(df)


def predictive_mean_matching(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Backwards-compatible wrapper for :class:`PMMImputer`."""
    return PMMImputer(k=k).impute(df)


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

