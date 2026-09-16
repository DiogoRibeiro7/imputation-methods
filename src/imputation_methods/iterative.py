"""Iterative multivariate imputers built on scikit-learn's IterativeImputer."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from ._utils import fit_transform_non_empty
from .base import BaseImputer
from .statistical import MeanImputer, MedianImputer

logger = logging.getLogger(__name__)


def _fit_transform_or_fallback(
    imputer: Any, df: pd.DataFrame, name: str, fallback: BaseImputer
) -> pd.DataFrame:
    """Run a scikit-learn imputer, falling back to ``fallback`` on data errors.

    Raises:
        RuntimeError: If the imputer fails for any reason other than a
            ``ValueError`` or ``LinAlgError``.
    """
    try:
        return fit_transform_non_empty(imputer, df)
    except (ValueError, np.linalg.LinAlgError) as e:
        logger.warning(
            "%s imputation failed (%s); falling back to %s",
            name,
            e,
            type(fallback).__name__,
        )
        return fallback.impute(df)
    except Exception as e:
        raise RuntimeError(f"{name} imputation failed: {e}") from e


class MICEImputer(BaseImputer):
    """Impute missing data using Multiple Imputation by Chained Equations (MICE).

    Each column with missing values is modelled as a regression on the other
    columns, cycling through the columns until the imputations stabilise.
    Returns a single completed dataset (scikit-learn's ``IterativeImputer``
    with its default ``BayesianRidge`` estimator).

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imputation_methods import MICEImputer
        >>> df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [2, 4, 6, np.nan]})
        >>> bool(MICEImputer(random_state=0).impute(df).notna().all().all())
        True
    """

    def __init__(self, random_state: int | None = None) -> None:
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
            Imputed dataframe. Falls back to mean imputation if the model cannot
            be fitted.

        Raises:
            RuntimeError: If MICE imputation fails unexpectedly.
        """
        df = self._ensure_numeric(df)
        return _fit_transform_or_fallback(self._imputer, df, "MICE", MeanImputer())


class EMImputer(BaseImputer):
    """Iterative, EM-style imputation.

    Alternates between fitting a model of each column given the others and
    re-imputing the missing entries, starting from mean imputation, until the
    imputations change by less than ``tol``.

    Note:
        This is not the closed-form EM for a multivariate normal distribution.
        It uses scikit-learn's ``IterativeImputer`` (chained equations with
        ``BayesianRidge``), which follows the same fit-then-impute loop. Use it
        when you want an iterative multivariate imputer with an explicit
        iteration budget and tolerance.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imputation_methods import EMImputer
        >>> df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, np.nan, 7, 8]})
        >>> imputed = EMImputer(max_iter=50, tol=1e-3, random_state=0).impute(df)
        >>> bool(imputed.notna().all().all())
        True

    References:
        Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood
        from incomplete data via the EM algorithm. Journal of the Royal
        Statistical Society: Series B, 39(1), 1-22.
    """

    def __init__(
        self, max_iter: int = 100, tol: float = 1e-4, random_state: int | None = None
    ) -> None:
        """Initialize the imputer.

        Args:
            max_iter: Maximum number of imputation rounds.
            tol: Convergence tolerance on the change of imputed values.
            random_state: Random seed for reproducibility.
        """
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values iteratively.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)
        imputer = IterativeImputer(
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            initial_strategy="mean",
        )
        return fit_transform_non_empty(imputer, df)


class MissForestImputer(BaseImputer):
    """Impute missing data using the MissForest algorithm.

    Approximates MissForest with scikit-learn's ``IterativeImputer`` using a
    ``RandomForestRegressor`` as the per-column estimator, which captures
    non-linear relationships and interactions.

    References:
        Stekhoven, D. J., & Bühlmann, P. (2012). MissForest - non-parametric
        missing value imputation for mixed-type data. Bioinformatics, 28(1),
        112-118.
    """

    def __init__(self, random_state: int | None = None) -> None:
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
            Imputed dataframe. Falls back to median imputation if the model
            cannot be fitted.

        Raises:
            RuntimeError: If MissForest imputation fails unexpectedly.
        """
        df = self._ensure_numeric(df)
        return _fit_transform_or_fallback(
            self._imputer, df, "MissForest", MedianImputer()
        )


class GAINImputer(BaseImputer):
    """Placeholder for Generative Adversarial Imputation Nets (GAIN).

    Warning:
        No adversarial network is trained. This class currently delegates to
        scikit-learn's ``IterativeImputer`` and produces the same results as
        :class:`MICEImputer`. It is kept so code written against this API keeps
        working, and may be replaced by a true GAIN implementation later.

    References:
        Yoon, J., Jordon, J., & van der Schaar, M. (2018). GAIN: Missing data
        imputation using generative adversarial nets. ICML.
    """

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
            Imputed dataframe. Falls back to median imputation if the model
            cannot be fitted.

        Raises:
            RuntimeError: If imputation fails unexpectedly.
        """
        df = self._ensure_numeric(df)
        return _fit_transform_or_fallback(self._imputer, df, "GAIN", MedianImputer())
