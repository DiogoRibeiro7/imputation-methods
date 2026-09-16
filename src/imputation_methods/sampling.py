"""Donor-based imputers that fill gaps with sampled or reference values."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._utils import observed_median
from .base import BaseImputer

ReferenceValues: TypeAlias = (
    dict[str, float | np.floating[Any] | NDArray[np.floating[Any]]] | pd.DataFrame
)
"""Reference data for :class:`ColdDeckImputer`: per-column values or a dataframe."""


class RandomSamplingImputer(BaseImputer):
    """Impute by randomly sampling from observed values.

    Similar to Hot Deck but without stratification. Preserves the
    empirical distribution of observed values.

    Args:
        random_state: Random seed for reproducibility. Default: None

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_methods import RandomSamplingImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, np.nan, 6]})
        >>> imputer = RandomSamplingImputer(random_state=42)
        >>> imputed = imputer.impute(df)
    """

    def __init__(self, random_state: int | None = None) -> None:
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
                        observed.to_numpy(), size=n_missing, replace=True
                    )
                    result.loc[missing_mask, column] = sampled_values

        return result


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

        group_indices = [group.index for _, group in df.groupby(self.stratify_cols)]
        for idx in group_indices:
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
                result.loc[idx[missing_mask.to_numpy()], column] = rng.choice(
                    donors.to_numpy(), size=missing_mask.sum(), replace=True
                )

        return result


class ColdDeckImputer(BaseImputer):
    """Cold deck imputation using predetermined reference values.

    Uses values from a reference dataset or predetermined mapping to fill
    missing values. Useful when you have historical or domain knowledge.

    Args:
        reference_values: Dictionary mapping column names to reference values
            or a reference DataFrame. If dict, can map to scalar or array.
            Default: None (uses column median as fallback)
        random_state: Seed used when sampling from array reference values.
            Default: None

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_methods import ColdDeckImputer
        >>> df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [np.nan, 2, 3]})
        >>> # Use predetermined values
        >>> imputer = ColdDeckImputer(reference_values={'a': 2.5, 'b': 2.0})
        >>> imputed = imputer.impute(df)

    References:
        Traditional imputation method predating hot deck imputation.
        Uses external/historical data rather than current dataset.
    """

    def __init__(
        self,
        reference_values: ReferenceValues | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initialize the cold deck imputer.

        Args:
            reference_values: Reference values for imputation.
            random_state: Seed used when sampling from array reference values.
        """
        self.reference_values = reference_values
        self.random_state = random_state

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using cold deck reference values.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        rng = np.random.default_rng(self.random_state)
        df = self._ensure_numeric(df)
        result = df.copy()

        if self.reference_values is None:
            # Fallback to median
            for column in result.columns:
                if result[column].isna().any():
                    result[column] = result[column].fillna(
                        observed_median(result[column])
                    )
        elif isinstance(self.reference_values, pd.DataFrame):
            # Use reference DataFrame
            for column in result.columns:
                if (
                    column in self.reference_values.columns
                    and result[column].isna().any()
                ):
                    ref_mean = self.reference_values[column].mean()
                    result[column] = result[column].fillna(ref_mean)
        elif isinstance(self.reference_values, dict):
            # Use dictionary of reference values
            for column in result.columns:
                if result[column].isna().any():
                    if column in self.reference_values:
                        ref_value = self.reference_values[column]
                        if isinstance(ref_value, (int, float)):
                            # Scalar reference value
                            result[column] = result[column].fillna(ref_value)
                        elif isinstance(ref_value, np.ndarray):
                            # Array reference value - sample randomly
                            fill_values = rng.choice(
                                ref_value, size=result[column].isna().sum()
                            )
                            result.loc[result[column].isna(), column] = fill_values
                        else:
                            # Numpy scalar types
                            result[column] = result[column].fillna(float(ref_value))
                    else:
                        # Fallback to median if column not in reference
                        result[column] = result[column].fillna(
                            observed_median(result[column])
                        )

        return result
