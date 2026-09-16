"""Abstract base class shared by every imputer."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ImputationError(RuntimeError):
    """An imputer's model could not be fitted and ``on_error="raise"`` was set."""


class BaseImputer(ABC):
    """Base class for all imputation methods.

    Subclasses implement :meth:`impute`, which must return a new dataframe and
    leave the input untouched.
    """

    def _ensure_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that ``df`` is a dataframe with only numeric columns.

        Args:
            df: Dataframe to validate.

        Returns:
            The original dataframe if all columns are numeric.

        Raises:
            TypeError: If ``df`` is not a dataframe or includes any non-numeric
                columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
        non_numeric = [
            column
            for column, dtype in df.dtypes.items()
            if not pd.api.types.is_numeric_dtype(dtype)
        ]
        if non_numeric:
            raise TypeError(f"All columns must be numeric; got {non_numeric}")
        return df

    @abstractmethod
    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in ``df``.

        Args:
            df: Dataframe with potential NaN values.

        Returns:
            Dataframe with imputed values.
        """
