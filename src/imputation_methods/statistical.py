"""Univariate statistical imputers.

Each column is filled independently from a summary statistic of its own
observed values (mean, median, mode, quantile, ...).
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from .base import BaseImputer


class MeanImputer(BaseImputer):
    """Impute missing values using column means.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_methods import MeanImputer
        >>> df = pd.DataFrame({"a": [1, 2, np.nan, 4]})
        >>> imputer = MeanImputer()
        >>> imputed = imputer.impute(df)
        >>> print(imputed.loc[2, "a"])  # Mean of [1, 2, 4]
        2.333...
    """

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill each column's missing values with that column's mean.

        Args:
            df: Dataframe with potential NaN values.

        Returns:
            Imputed dataframe. Columns with no observed values stay NaN.
        """
        df = self._ensure_numeric(df)
        result = df.copy()
        for column in result.columns:
            mean_val = result[column].mean()
            result[column] = result[column].fillna(mean_val)

        return result


class MedianImputer(BaseImputer):
    """Impute missing values using column medians.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_methods import MedianImputer
        >>> df = pd.DataFrame({"a": [1, 2, np.nan, 10]})
        >>> imputer = MedianImputer()
        >>> imputed = imputer.impute(df)
        >>> print(imputed.loc[2, "a"])  # Median of [1, 2, 10]
        2.0
    """

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill each column's missing values with that column's median.

        Args:
            df: Dataframe with potential NaN values.

        Returns:
            Imputed dataframe. Columns with no observed values stay NaN.
        """
        df = self._ensure_numeric(df)
        result = df.copy()
        for column in result.columns:
            median_val = result[column].median()
            result[column] = result[column].fillna(median_val)
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
        >>> from imputation_methods import ModeImputer
        >>> df = pd.DataFrame({'a': [1, 2, 2, np.nan, 2, 3]})
        >>> imputer = ModeImputer()
        >>> imputed = imputer.impute(df)
        >>> # Missing value filled with 2 (most frequent)

    References:
        Standard statistical technique for categorical/discrete data.
    """

    def __init__(self, dropna: bool = True) -> None:
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
        >>> from imputation_methods import ConstantImputer
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

    def __init__(self, fill_value: float | dict[str, float] = 0) -> None:
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
                raise ValueError(f"fill_value contains unknown columns: {unknown_cols}")

            # Fill each column with its specific value
            for column, value in self.fill_value.items():
                if column in result.columns and result[column].isna().any():
                    result[column] = result[column].fillna(value)
        else:
            # Fill all columns with the same value
            result = result.fillna(self.fill_value)

        return result


class QuantileImputer(BaseImputer):
    """Impute using specified quantile of observed values.

    Args:
        quantile: Quantile to use (0.0 to 1.0). Default: 0.5 (median)

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_methods import QuantileImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, 5]})
        >>> # Use 75th percentile
        >>> imputer = QuantileImputer(quantile=0.75)
        >>> imputed = imputer.impute(df)
    """

    def __init__(self, quantile: float = 0.5) -> None:
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


class TrimmedMeanImputer(BaseImputer):
    """Trimmed mean imputation excluding extreme values.

    Computes mean after removing a percentage of extreme values
    from both ends. More robust than simple mean.

    Args:
        trim_fraction: Fraction to trim from each end (0-0.5). Default: 0.1

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_methods import TrimmedMeanImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, 100]})  # 100 is outlier
        >>> imputer = TrimmedMeanImputer(trim_fraction=0.2)
        >>> imputed = imputer.impute(df)
        >>> # Excludes 100 from mean calculation

    References:
        Robust statistics using trimmed estimators.
    """

    def __init__(self, trim_fraction: float = 0.1) -> None:
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
                    result[column].dropna(), self.trim_fraction
                )
                result[column] = result[column].fillna(trimmed_mean)

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
        >>> from imputation_methods import EndOfDistributionImputer
        >>> df = pd.DataFrame({'a': [1, 2, 3, np.nan, 5]})
        >>> imputer = EndOfDistributionImputer(position='high', k=2)
        >>> imputed = imputer.impute(df)
        >>> # Missing value filled with mean + 2*std

    References:
        Used in outlier detection and robust imputation strategies.
    """

    def __init__(self, position: str = "high", k: float = 3.0) -> None:
        """Initialize the end-of-distribution imputer.

        Args:
            position: 'low' (mean - k*std) or 'high' (mean + k*std)
            k: Number of standard deviations

        Raises:
            ValueError: If position is not 'low' or 'high'
        """
        if position not in ["low", "high"]:
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

                if self.position == "low":
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
        >>> from imputation_methods import GroupMeanImputer
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
        self, group_col: str, method: str = "mean", global_fallback: bool = True
    ) -> None:
        """Initialize the group mean imputer.

        Args:
            group_col: Column name to group by
            method: 'mean' or 'median'
            global_fallback: Use global statistic for groups with no data

        Raises:
            ValueError: If method is not 'mean' or 'median'
        """
        if method not in ["mean", "median"]:
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
            col
            for col in result.columns
            if col != self.group_col and pd.api.types.is_numeric_dtype(result[col])
        ]

        for column in numeric_cols:
            if result[column].isna().any():
                grouped = result.groupby(self.group_col)[column]
                group_stats = (
                    grouped.transform("mean")
                    if self.method == "mean"
                    else grouped.transform("median")
                )
                result[column] = result[column].fillna(group_stats)

                # Handle any remaining NaNs with global fallback
                if self.global_fallback and result[column].isna().any():
                    if self.method == "mean":
                        global_stat = df[column].mean()
                    else:  # median
                        global_stat = df[column].median()

                    result[column] = result[column].fillna(global_stat)

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
        >>> from imputation_methods import IndicatorImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4], 'b': [5, np.nan, 7, 8]})
        >>> imputer = IndicatorImputer(strategy='mean')
        >>> imputed = imputer.impute(df)
        >>> print(imputed.columns.tolist())
        ['a', 'b', 'missing_a', 'missing_b']
    """

    def __init__(
        self, strategy: str = "mean", indicator_prefix: str = "missing_"
    ) -> None:
        """Initialize the indicator imputer.

        Args:
            strategy: Imputation strategy
            indicator_prefix: Prefix for indicator columns
        """
        if strategy not in ["mean", "median", "zero"]:
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
                if self.strategy == "mean":
                    fill_value = result[column].mean()
                elif self.strategy == "median":
                    fill_value = result[column].median()
                else:  # zero
                    fill_value = 0

                result[column] = result[column].fillna(fill_value)

        return result
