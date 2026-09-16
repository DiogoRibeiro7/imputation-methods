"""Neural-network imputers."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from .base import BaseImputer
from .statistical import MeanImputer

logger = logging.getLogger(__name__)


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

        Raises:
            RuntimeError: If autoencoder training or prediction fails.
        """
        df = self._ensure_numeric(df)
        try:
            filled = df.fillna(df.mean())
            # Handle case where mean might be NaN (all values missing)
            if filled.isna().any().any():
                filled = filled.fillna(0)
            self._model.fit(filled, filled)
            reconstructed = pd.DataFrame(
                self._model.predict(filled),
                columns=df.columns,
                index=df.index,
            )
            return df.where(~df.isna(), reconstructed)
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.warning(
                "Autoencoder imputation failed (%s); falling back to mean imputation",
                e,
            )
            return MeanImputer().impute(df)
        except Exception as e:
            raise RuntimeError(f"Autoencoder imputation failed: {e}") from e
