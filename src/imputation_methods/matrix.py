"""Low-rank matrix completion imputers.

Both algorithms are implemented directly on NumPy so the package does not depend
on unmaintained third-party solvers.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._deprecation import renamed_module_attributes, renamed_parameters
from ._utils import OnError, check_on_error, raise_or_fall_back
from .base import BaseImputer
from .statistical import MeanImputer

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

_INIT_FILL_METHODS = ("zero", "mean", "median", "min")
_SIGMA2_FLOOR = 1e-6


def _initial_fill(
    values: FloatArray, missing: NDArray[np.bool_], method: str
) -> FloatArray:
    """Return a copy of ``values`` with missing entries filled column-wise."""
    filled = values.copy()
    if method == "zero":
        filled[missing] = 0.0
        return filled
    if method == "mean":
        column_stats = np.nanmean(values, axis=0)
    elif method == "median":
        column_stats = np.nanmedian(values, axis=0)
    else:
        column_stats = np.nanmin(values, axis=0)
    rows, cols = np.nonzero(missing)
    filled[rows, cols] = column_stats[cols]
    return filled


def _soft_impute(
    values: FloatArray,
    *,
    shrinkage_value: float | None,
    max_iter: int,
    convergence_threshold: float,
    init_fill_method: str,
) -> FloatArray:
    """Complete ``values`` by iterative soft-thresholded SVD.

    Implements the SoftImpute algorithm of Mazumder, Hastie & Tibshirani (2010):
    repeatedly replace the missing entries with those of a low-rank
    reconstruction whose singular values are shrunk by ``shrinkage_value``.
    """
    missing = np.isnan(values)
    filled = _initial_fill(values, missing, init_fill_method)
    if not missing.any():
        return filled

    if shrinkage_value is None:
        # Keep only components carrying at least 1/50 of the leading singular
        # value of the initial fill.
        shrinkage_value = float(np.linalg.norm(filled, ord=2)) / 50.0

    for _ in range(max_iter):
        u, s, vt = np.linalg.svd(filled, full_matrices=False)
        s_shrunk = np.maximum(s - shrinkage_value, 0.0)
        reconstruction = (u * s_shrunk) @ vt

        previous = filled[missing]
        current = reconstruction[missing]
        filled[missing] = current

        previous_norm = np.linalg.norm(previous)
        if previous_norm > 0:
            change = np.linalg.norm(current - previous) / previous_norm
            if change < convergence_threshold:
                break
    return filled


def _ppca_impute(
    values: FloatArray, *, n_components: int, max_iter: int, tol: float
) -> FloatArray:
    """Complete standardized ``values`` with probabilistic PCA fitted by EM.

    Follows the EM updates of Tipping & Bishop (1999). After each iteration the
    missing entries are replaced by their expected value under the current
    model, so the fit and the imputations converge together.
    """
    n_samples, n_features = values.shape
    missing = np.isnan(values)
    data = np.where(missing, 0.0, values)
    identity = np.eye(n_components)

    # Deterministic initialisation from the leading principal directions.
    mean = data.mean(axis=0)
    _, s, vt = np.linalg.svd(data - mean, full_matrices=False)
    loadings = vt[:n_components].T * (s[:n_components] / np.sqrt(n_samples))
    sigma2 = 1.0

    for _ in range(max_iter):
        mean = data.mean(axis=0)
        centered = data - mean

        # E-step: posterior moments of the latent variables.
        m_inv = np.linalg.inv(loadings.T @ loadings + sigma2 * identity)
        latent = centered @ loadings @ m_inv
        latent_outer = n_samples * sigma2 * m_inv + latent.T @ latent

        # M-step: update loadings and isotropic noise variance.
        loadings = np.linalg.solve(latent_outer, latent.T @ centered).T
        sigma2 = (
            np.sum(centered**2)
            - 2.0 * np.sum(latent * (centered @ loadings))
            + np.trace(latent_outer @ (loadings.T @ loadings))
        ) / (n_samples * n_features)
        sigma2 = max(float(sigma2), _SIGMA2_FLOOR)

        # Re-impute missing entries with the model's conditional expectation.
        m_inv = np.linalg.inv(loadings.T @ loadings + sigma2 * identity)
        reconstruction = centered @ loadings @ m_inv @ loadings.T + mean

        previous = data[missing]
        current = reconstruction[missing]
        data[missing] = current

        previous_norm = np.linalg.norm(previous)
        if previous_norm > 0 and np.linalg.norm(current - previous) < (
            tol * previous_norm
        ):
            break
    return data


class SoftImputeImputer(BaseImputer):
    """Impute missing values by low-rank matrix completion (SoftImpute).

    Missing entries are initialised with ``init_fill_method`` and then
    iteratively replaced by a low-rank SVD reconstruction whose singular values
    are soft-thresholded, which is equivalent to nuclear-norm regularisation.
    Columns with no observed values are left untouched.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imputation_methods import SoftImputeImputer
        >>> df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
        >>> bool(SoftImputeImputer().impute(df).notna().all().all())
        True

    References:
        Mazumder, R., Hastie, T., & Tibshirani, R. (2010). Spectral
        regularization algorithms for learning large incomplete matrices.
        Journal of Machine Learning Research, 11, 2287-2322.
    """

    @renamed_parameters(max_iters="max_iter")
    def __init__(
        self,
        max_iter: int = 100,
        init_fill_method: str = "zero",
        shrinkage_value: float | None = None,
        convergence_threshold: float = 1e-3,
        on_error: OnError = None,
    ) -> None:
        """Initialize the imputer.

        Args:
            max_iter: Maximum number of SVD iterations.
            init_fill_method: How to initialise missing entries before solving:
                ``"zero"``, ``"mean"``, ``"median"`` or ``"min"``.
            shrinkage_value: Amount subtracted from each singular value. Defaults
                to 1/50 of the largest singular value of the initial fill.
            convergence_threshold: Stop when the relative change of the imputed
                entries between iterations falls below this value.
            on_error: What to do if the model can't be fitted: ``"raise"`` an
                :class:`~imputation_methods.ImputationError`, or ``"fallback"``
                to use mean imputation. The default, ``None``, falls back with a
                ``FutureWarning``; it will change to ``"raise"`` in 1.0.0.

        Raises:
            ValueError: If an argument is out of range.
        """
        if init_fill_method not in _INIT_FILL_METHODS:
            raise ValueError(
                f"init_fill_method must be one of {_INIT_FILL_METHODS}, "
                f"got {init_fill_method!r}"
            )
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if shrinkage_value is not None and shrinkage_value < 0:
            raise ValueError(f"shrinkage_value must be >= 0, got {shrinkage_value}")
        self.max_iter = max_iter
        self.init_fill_method = init_fill_method
        self.shrinkage_value = shrinkage_value
        self.convergence_threshold = convergence_threshold
        self.on_error = check_on_error(on_error)

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using a low-rank matrix approximation.

        Args:
            df: Dataframe with potential NaN values.

        Returns:
            Dataframe with missing entries imputed by SoftImpute.
        """
        df = self._ensure_numeric(df)
        result = df.astype(float)
        modelled = result.columns[result.notna().any()]
        if modelled.empty or not result[modelled].isna().any().any():
            return result

        try:
            completed = _soft_impute(
                result[modelled].to_numpy(dtype=float, na_value=np.nan),
                shrinkage_value=self.shrinkage_value,
                max_iter=self.max_iter,
                convergence_threshold=self.convergence_threshold,
                init_fill_method=self.init_fill_method,
            )
        except np.linalg.LinAlgError as e:
            raise_or_fall_back(
                self.on_error,
                imputer=type(self).__name__,
                error=e,
                fallback="mean imputation",
            )
            return MeanImputer().impute(df)

        result[modelled] = completed
        return result


class PPCAImputer(BaseImputer):
    """Impute missing values with probabilistic PCA (PPCA).

    Columns are standardised, then a PPCA model is fitted by EM while the
    missing entries are repeatedly replaced by their expected value under the
    model. The model is the maximum-likelihood PPCA of Tipping & Bishop; no
    priors are placed on the loadings.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imputation_methods import PPCAImputer
        >>> df = pd.DataFrame(
        ...     {"a": [1.0, 2.0, np.nan, 4.0], "b": [2.0, np.nan, 6.0, 8.0]}
        ... )
        >>> bool(PPCAImputer().impute(df).notna().all().all())
        True

    References:
        Tipping, M. E., & Bishop, C. M. (1999). Probabilistic principal
        component analysis. Journal of the Royal Statistical Society: Series B,
        61(3), 611-622.
    """

    def __init__(
        self,
        n_components: int | None = 1,
        min_obs: int = 1,
        max_iter: int = 500,
        tol: float = 1e-6,
        on_error: OnError = None,
    ) -> None:
        """Initialize the imputer.

        Args:
            n_components: Number of latent dimensions. ``None`` uses 2. The value
                is capped at ``n_columns - 1`` so the model stays low-rank.
            min_obs: Minimum number of observed values a column needs to take
                part in the model. Other columns are mean-imputed.
            max_iter: Maximum number of EM iterations.
            tol: Stop when the relative change of the imputed entries between
                iterations falls below this value.
            on_error: What to do if the model can't be fitted: ``"raise"`` an
                :class:`~imputation_methods.ImputationError`, or ``"fallback"``
                to use mean imputation. The default, ``None``, falls back with a
                ``FutureWarning``; it will change to ``"raise"`` in 1.0.0.

        Raises:
            ValueError: If an argument is out of range.
        """
        if n_components is not None and n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if min_obs < 1:
            raise ValueError(f"min_obs must be >= 1, got {min_obs}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        self.n_components = n_components
        self.min_obs = min_obs
        self.max_iter = max_iter
        self.tol = tol
        self.on_error = check_on_error(on_error)

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using a probabilistic PCA model.

        Args:
            df: Dataframe with missing values.

        Returns:
            Dataframe with missing entries filled via probabilistic PCA. A
            single-column dataframe is returned unchanged, since PPCA needs at
            least two columns.
        """
        df = self._ensure_numeric(df)
        if df.shape[1] == 1:
            logger.warning("PPCA needs at least two columns; returning input copy")
            return df.copy()

        result = df.astype(float)
        enough_obs = result.notna().sum() >= self.min_obs
        modelled = result.columns[enough_obs]
        if len(modelled) < 2:
            logger.warning(
                "Fewer than two columns have %d observations; using mean imputation",
                self.min_obs,
            )
            return MeanImputer().impute(df)

        values = result[modelled].to_numpy(dtype=float, na_value=np.nan)
        means = np.nanmean(values, axis=0)
        stds = np.nanstd(values, axis=0)
        stds[~(stds > 0)] = 1.0
        requested = 2 if self.n_components is None else self.n_components
        n_components = min(requested, len(modelled) - 1)

        try:
            completed = _ppca_impute(
                (values - means) / stds,
                n_components=n_components,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        except np.linalg.LinAlgError as e:
            raise_or_fall_back(
                self.on_error,
                imputer=type(self).__name__,
                error=e,
                fallback="mean imputation",
            )
            return MeanImputer().impute(df)

        result[modelled] = completed * stds + means
        excluded = result.columns[~enough_obs]
        if len(excluded) > 0:
            result[excluded] = MeanImputer().impute(result[excluded])
        return result


__getattr__ = renamed_module_attributes(__name__)
