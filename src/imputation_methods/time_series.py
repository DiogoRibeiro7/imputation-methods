"""Imputers that exploit row order, for time series and sequential data."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression,
)

from .base import BaseImputer


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


class ForwardFillFallbackImputer(BaseImputer):
    """Forward fill with fallback to mean/median for leading NaNs.

    Combines LOCF with a fallback strategy for initial missing values
    that cannot be forward filled.

    Args:
        fallback: Fallback strategy ('mean' or 'median'). Default: 'mean'

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_methods import ForwardFillFallbackImputer
        >>> df = pd.DataFrame({'a': [np.nan, np.nan, 3, np.nan, 5]})
        >>> imputer = ForwardFillFallbackImputer(fallback='mean')
        >>> imputed = imputer.impute(df)
        >>> # First two NaNs filled with mean, third NaN forward filled
    """

    def __init__(self, fallback: str = "mean") -> None:
        """Initialize the forward fill fallback imputer.

        Args:
            fallback: Fallback strategy ('mean' or 'median')

        Raises:
            ValueError: If fallback is not 'mean' or 'median'
        """
        if fallback not in ["mean", "median"]:
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
                result[column] = result[column].ffill()

                # Then fill remaining NaNs with fallback
                if result[column].isna().any():
                    if self.fallback == "mean":
                        fill_value = result[column].mean()
                    else:  # median
                        fill_value = result[column].median()

                    result[column] = result[column].fillna(fill_value)

        return result


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
        >>> from imputation_methods import InterpolationImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, np.nan, 5]})
        >>> imputer = InterpolationImputer(method='linear')
        >>> imputed = imputer.impute(df)
        >>> print(imputed['a'].tolist())
        [1.0, 2.0, 3.0, 4.0, 5.0]
    """

    def __init__(
        self,
        method: str = "linear",
        order: int = 2,
        limit: int | None = None,
        limit_direction: Literal["forward", "backward", "both"] = "both",
    ) -> None:
        """Initialize the interpolation imputer.

        Args:
            method: Interpolation method
            order: Polynomial/spline order
            limit: Maximum consecutive NaNs to fill
            limit_direction: Fill direction
        """
        valid_methods = ["linear", "polynomial", "spline"]
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
                if self.method == "linear":
                    result[column] = result[column].interpolate(
                        method="linear",
                        limit=self.limit,
                        limit_direction=self.limit_direction,
                    )
                elif self.method == "polynomial":
                    result[column] = result[column].interpolate(
                        method="polynomial",
                        order=self.order,
                        limit=self.limit,
                        limit_direction=self.limit_direction,
                    )
                elif self.method == "spline":
                    result[column] = result[column].interpolate(
                        method="spline",
                        order=self.order,
                        limit=self.limit,
                        limit_direction=self.limit_direction,
                    )

                # Fill any remaining NaNs with forward/backward fill
                result[column] = result[column].ffill()
                result[column] = result[column].bfill()

        return result


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
        >>> from imputation_methods import MovingAverageImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, np.nan, 6]})
        >>> imputer = MovingAverageImputer(window=3, method='mean')
        >>> imputed = imputer.impute(df)
    """

    def __init__(
        self,
        window: int = 3,
        method: str = "mean",
        min_periods: int = 1,
        center: bool = False,
    ) -> None:
        """Initialize the moving average imputer.

        Args:
            window: Window size
            method: 'mean' or 'median'
            min_periods: Minimum observations required
            center: Center the window
        """
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if method not in ["mean", "median"]:
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
                    window=self.window, min_periods=self.min_periods, center=self.center
                )

                if self.method == "mean":
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
        >>> from imputation_methods import WeightedMovingAverageImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, np.nan, 6]})
        >>> imputer = WeightedMovingAverageImputer(alpha=0.7)
        >>> imputed = imputer.impute(df)
        >>> # Missing values filled using exponentially weighted average

    References:
        Commonly used in financial time series and sensor data analysis.
    """

    def __init__(self, alpha: float = 0.5, min_periods: int = 1) -> None:
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
                ewma = (
                    result[column]
                    .ewm(alpha=self.alpha, min_periods=self.min_periods, ignore_na=True)
                    .mean()
                )

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
        >>> from imputation_methods import LinearTrendImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, np.nan, 6]})
        >>> imputer = LinearTrendImputer()
        >>> imputed = imputer.impute(df)
        >>> # Missing values filled based on linear trend

    References:
        Standard technique for trending time series data.
    """

    def __init__(self, use_index: bool = False) -> None:
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
                observed_values = result.loc[mask, column].to_numpy(dtype=float)

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
                    result.loc[result[column].isna(), column] = predictions[
                        result[column].isna()
                    ]

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
        >>> from imputation_methods import PolynomialTrendImputer
        >>> df = pd.DataFrame({'a': [1, 4, np.nan, 16, np.nan, 36]})
        >>> imputer = PolynomialTrendImputer(degree=2)
        >>> imputed = imputer.impute(df)
        >>> # Missing values filled based on quadratic trend

    References:
        Used for time series with non-linear but smooth trends.
    """

    def __init__(self, degree: int = 2, use_index: bool = False) -> None:
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
                observed_values = result.loc[mask, column].to_numpy(dtype=float)

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
                            observed_positions, observed_values, effective_degree
                        )
                        predictions = np.polyval(coefficients, all_positions)

                    # Fill missing values
                    result.loc[result[column].isna(), column] = predictions[
                        result[column].isna()
                    ]

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
        >>> from imputation_methods import SeasonalImputer
        >>> # Daily data with weekly seasonality
        >>> df = pd.DataFrame({'sales': [100, 120, np.nan, 140, 130, np.nan, 90]})
        >>> imputer = SeasonalImputer(period=7, method='median')
        >>> imputed = imputer.impute(df)
    """

    def __init__(self, period: int = 7, method: str = "median") -> None:
        """Initialize the seasonal imputer.

        Args:
            period: Seasonal period
            method: 'mean' or 'median'
        """
        if period < 2:
            raise ValueError(f"period must be >= 2, got {period}")
        if method not in ["mean", "median"]:
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
                # Average each phase of the cycle over its observed values.
                phases = pd.Series(
                    np.arange(len(result)) % self.period, index=result.index
                )
                phase_stats = result[column].groupby(phases).agg(self.method)
                result[column] = result[column].fillna(phases.map(phase_stats))

                # Fill any remaining NaNs with overall mean
                if result[column].isna().any():
                    result[column] = result[column].fillna(result[column].mean())

        return result


class KalmanFilterImputer(BaseImputer):
    """Kalman filter imputation for time series with uncertainty.

    Uses Kalman filtering to impute missing values while accounting for
    measurement noise and process uncertainty. Ideal for sensor data.

    **Algorithm Overview:**
    The Kalman filter is a recursive Bayesian estimator that operates in
    two steps:
    1. **Prediction**: Estimates the next state based on the previous state
    2. **Update**: Corrects the prediction using new measurements

    For imputation, when a measurement is missing, we use only the prediction
    step to fill the gap.

    **Mathematical Background:**
    - State equation: x_k = x_{k-1} + w_k, where w_k ~ N(0, Q)
    - Measurement equation: z_k = x_k + v_k, where v_k ~ N(0, R)
    - Prediction: x_pred = x_est, P_pred = P_est + Q
    - Update: K = P_pred/(P_pred + R), x_est = x_pred + K*(z - x_pred)

    Args:
        process_variance: Process noise variance (Q). Controls how much the
            state can vary between time steps. Larger values allow more
            flexibility but may lead to overfitting. Default: 1.0
        measurement_variance: Measurement noise variance (R). Reflects
            confidence in observations. Larger values trust predictions more
            than measurements. Default: 1.0
        initial_state: Initial state estimate. If None, uses first observed
            value. Default: None
        initial_covariance: Initial error covariance (P). Represents initial
            uncertainty in state estimate. Default: 1.0

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_methods import KalmanFilterImputer
        >>> df = pd.DataFrame({'a': [1, np.nan, 3, np.nan, 5]})
        >>> # High process variance allows more variation
        >>> imputer = KalmanFilterImputer(process_variance=2.0)
        >>> imputed = imputer.impute(df)
        >>> # Missing values filled using Kalman filter estimates

    References:
        Kalman, R. E. (1960). A new approach to linear filtering and prediction.
        Journal of Basic Engineering, 82(1), 35-45.
        Widely used in sensor fusion, GPS, and state estimation.

    Notes:
        - Works best for time series data with smooth trends
        - Assumes linear state transitions (constant velocity model)
        - For non-linear systems, consider Extended Kalman Filter (EKF)
    """

    def __init__(
        self,
        process_variance: float = 1.0,
        measurement_variance: float = 1.0,
        initial_state: float | None = None,
        initial_covariance: float = 1.0,
    ) -> None:
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

        The Kalman filter processes data sequentially, maintaining an estimate
        of the current state and its uncertainty. At each time step:

        1. Predict the next state using the state transition model
        2. If a measurement exists, update the estimate using Kalman gain
        3. If no measurement exists (NaN), use the prediction as the imputed value

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

                # ============================================================
                # STEP 1: Initialize Kalman filter state
                # ============================================================
                # Find first observed value to initialize the filter
                first_obs_idx = np.where(~np.isnan(values))[0]
                if len(first_obs_idx) == 0:
                    continue  # No observed values in this column

                # Initialize state estimate (x_est)
                # This is our best guess of the true value at time k
                if self.initial_state is None:
                    x_est = values[first_obs_idx[0]]  # Use first observation
                else:
                    x_est = self.initial_state

                # Initialize error covariance (P_est)
                # This quantifies our uncertainty in the state estimate
                p_est = self.initial_covariance

                # ============================================================
                # STEP 2: Run Kalman filter through all time steps
                # ============================================================
                for i in range(len(values)):
                    # --------------------------------------------------------
                    # PREDICTION STEP
                    # --------------------------------------------------------
                    # Predict next state using state transition model
                    # For this simple model: x_k = x_{k-1} (constant velocity)
                    x_pred = x_est

                    # Predict error covariance
                    # Uncertainty grows by process_variance at each step
                    p_pred = p_est + self.process_variance

                    # --------------------------------------------------------
                    # UPDATE STEP (if measurement available)
                    # --------------------------------------------------------
                    if not np.isnan(values[i]):
                        # We have a measurement! Update our estimate

                        # Compute Kalman gain (K)
                        # K determines how much we trust the measurement vs prediction
                        # K → 1: trust measurement more (low measurement_variance)
                        # K → 0: trust prediction more (high measurement_variance)
                        kalman_gain = p_pred / (p_pred + self.measurement_variance)

                        # Update state estimate using measurement
                        # x_est = prediction + gain * (measurement - prediction)
                        # This is a weighted average of prediction and measurement
                        x_est = x_pred + kalman_gain * (values[i] - x_pred)

                        # Update error covariance
                        # Uncertainty decreases when we incorporate a measurement
                        p_est = (1 - kalman_gain) * p_pred
                    else:
                        # --------------------------------------------------------
                        # NO MEASUREMENT (NaN) - Use prediction for imputation
                        # --------------------------------------------------------
                        # Fill missing value with predicted state
                        values[i] = x_pred

                        # State estimate remains at prediction
                        x_est = x_pred

                        # Error covariance remains at predicted value
                        # (uncertainty doesn't decrease without measurement)
                        p_est = p_pred

                # Update column with imputed values
                result[column] = values

        return result
