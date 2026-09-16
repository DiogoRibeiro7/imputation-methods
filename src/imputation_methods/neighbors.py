"""Distance-based imputers that borrow values from similar rows."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer as _SklearnKNNImputer
from sklearn.neighbors import RadiusNeighborsRegressor

from ._deprecation import renamed_module_attributes, renamed_parameters
from ._utils import fit_transform_non_empty
from .base import BaseImputer
from .statistical import MeanImputer

logger = logging.getLogger(__name__)


class KNNImputer(BaseImputer):
    """Impute missing values using K-nearest neighbors.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_methods import KNNImputer
        >>> df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, np.nan, 7, 8]})
        >>> imputer = KNNImputer(n_neighbors=2)
        >>> imputed = imputer.impute(df)
        >>> assert not imputed.isna().any().any()
    """

    @renamed_parameters(k="n_neighbors")
    def __init__(self, n_neighbors: int = 5) -> None:
        """Initialize the imputer.

        Args:
            n_neighbors: Number of neighbors to consider.

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
        self._imputer = _SklearnKNNImputer(n_neighbors=n_neighbors)

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using the fitted KNN strategy.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.

        Raises:
            ValueError: If dataframe is empty or has insufficient data for KNN.
            RuntimeError: If KNN imputation fails.
        """
        df = self._ensure_numeric(df)
        if not df.isna().any().any():
            return df.copy()

        try:
            return fit_transform_non_empty(self._imputer, df)
        except ValueError as e:
            logger.warning(
                "KNN imputation failed (%s); falling back to mean imputation", e
            )
            return MeanImputer().impute(df)
        except Exception as e:
            raise RuntimeError(f"KNN imputation failed: {e}") from e


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
        >>> from imputation_methods import RadiusNeighborsImputer
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
        self, radius: float = 1.0, weights: str = "distance", metric: str = "euclidean"
    ) -> None:
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
                            radius=self.radius, weights=self.weights, metric=self.metric
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
        >>> from imputation_methods import LocalMeanImputer
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, 5]})
        >>> imputer = LocalMeanImputer(n_neighbors=3)
        >>> imputed = imputer.impute(df)

    References:
        Locally weighted averaging for smooth imputation.
    """

    def __init__(
        self, n_neighbors: int = 5, distance_weight_power: float = 2.0
    ) -> None:
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

                    complete_features = (
                        result.loc[complete_mask, feature_cols].fillna(0).values
                    )
                    complete_values = result.loc[complete_mask, column].to_numpy(
                        dtype=float
                    )

                    # Compute Euclidean distances
                    distances = np.sqrt(
                        np.sum((complete_features - row_features) ** 2, axis=1)
                    )

                    # Get k nearest neighbors
                    k = min(self.n_neighbors, len(distances))
                    nearest_idx = np.argsort(distances)[:k]

                    # Compute weights (inverse distance)
                    nearest_distances = distances[nearest_idx]
                    # Avoid division by zero
                    nearest_distances = np.maximum(nearest_distances, 1e-10)
                    weights = 1.0 / (nearest_distances**self.distance_weight_power)
                    weights /= weights.sum()

                    # Weighted average
                    result.loc[idx, column] = np.sum(
                        weights * complete_values[nearest_idx]
                    )

        return result


__getattr__ = renamed_module_attributes(__name__)
