"""Meta-imputers that combine or chain other imputers."""

from __future__ import annotations

import copy
import inspect
import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseImputer
from .statistical import MeanImputer, MedianImputer
from .time_series import InterpolationImputer

logger = logging.getLogger(__name__)


def _mean_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Element-wise, label-aligned mean of equally shaped dataframes."""
    total = frames[0]
    for frame in frames[1:]:
        total = total + frame
    return total / len(frames)


class HybridImputer(BaseImputer):
    """Hybrid imputation combining multiple methods with fallback chain.

    Tries multiple imputation methods in sequence, falling back to simpler
    methods if earlier methods fail or produce NaNs. Robust for diverse data.

    **Strategy Pattern:**
    This imputer implements a cascading fallback strategy where sophisticated
    methods are tried first, with progressively simpler methods as fallbacks.
    This approach combines the advantages of multiple methods while ensuring
    robustness.

    **Use Cases:**
    - **Heterogeneous data**: Different columns may need different approaches
    - **Unknown data patterns**: Not sure which method will work best
    - **Production systems**: Need guaranteed imputation without failures
    - **Exploratory analysis**: Want to leverage multiple strategies

    **Design Principles:**
    1. **Graceful degradation**: Complex methods → Simple methods → Always succeed
    2. **Error resilience**: Method failures don't crash the pipeline
    3. **Early stopping**: Stop once all NaNs are filled (efficiency)
    4. **Guaranteed completion**: Final fallback ensures no NaNs remain

    **Recommended Method Ordering:**
    1. **Domain-specific** (if applicable): Business rules, external data
    2. **Sophisticated**: ML-based (MICE, MissForest, etc.)
    3. **Moderate**: Statistical (interpolation, regression)
    4. **Simple**: Basic stats (mean, median)

    Args:
        methods: List of imputer instances to try in order. Methods should be
            ordered from most sophisticated/specific to simplest/most general.
            Default: [InterpolationImputer(), MeanImputer()]

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_methods import (
        ...     HybridImputer, InterpolationImputer,
        ...     MovingAverageImputer, MeanImputer
        ... )
        >>> df = pd.DataFrame({'a': [1, np.nan, np.nan, 4, np.nan]})
        >>> # Cascade: interpolation → moving average → mean
        >>> imputer = HybridImputer(methods=[
        ...     InterpolationImputer(),      # Try smooth interpolation first
        ...     MovingAverageImputer(window=2),  # Fall back to local average
        ...     MeanImputer()                 # Final fallback: global mean
        ... ])
        >>> imputed = imputer.impute(df)

    Notes:
        - Each method sees the output of the previous method
        - If a method fills all NaNs, subsequent methods are skipped
        - Exceptions in individual methods are caught and logged
        - Always has a final safety net (mean → 0) to guarantee no NaNs

    References:
        Ensemble and cascading strategies for robust machine learning.
    """

    def __init__(self, methods: list[BaseImputer] | None = None) -> None:
        """Initialize the hybrid imputer.

        Args:
            methods: List of imputer instances to try in order
        """
        if methods is None:
            # Default fallback chain: interpolation → mean
            # Interpolation works well for smooth trends
            # Mean is a safe universal fallback
            methods = [InterpolationImputer(), MeanImputer()]

        self.methods = methods

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using hybrid fallback chain.

        **Execution Flow:**
        1. Start with original data
        2. For each method in the chain:
           - Check if NaNs remain
           - If yes: try the method
           - If method succeeds: use its output
           - If method fails: log warning and try next
           - If no NaNs remain: stop (early exit)
        3. Final safety check: mean fallback for any remaining NaNs

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe (guaranteed to have no NaNs).
        """
        df = self._ensure_numeric(df)
        result = df.copy()

        # ============================================================
        # MAIN FALLBACK CHAIN
        # ============================================================
        # Try each method sequentially until all NaNs are filled
        for method in self.methods:
            # Check if there are still NaNs to fill
            if result.isna().any().any():
                try:
                    # Attempt imputation with current method
                    result = method.impute(result)

                    # Log successful imputation (useful for debugging)
                    logger.debug(
                        "%s imputed data; %d NaNs remain",
                        type(method).__name__,
                        result.isna().sum().sum(),
                    )

                except Exception as e:
                    # Method failed - log warning and continue to next method
                    # This ensures pipeline robustness
                    logger.warning(
                        "%s failed (%s); trying the next method in the chain",
                        type(method).__name__,
                        e,
                    )
                    continue
            else:
                # ========================================================
                # EARLY EXIT: All NaNs filled
                # ========================================================
                # No need to try remaining methods
                # This saves computation time
                logger.debug("All NaNs filled. Skipping remaining methods.")
                break

        # ============================================================
        # FINAL SAFETY NET
        # ============================================================
        # Guarantee no NaNs remain, even if all methods failed
        # This ensures the imputer always succeeds
        if result.isna().any().any():
            logger.warning(
                "Some NaNs remain after all methods. Applying final fallback."
            )

            for column in result.columns:
                if result[column].isna().any():
                    # Try mean first (most reasonable fallback)
                    mean_val = result[column].mean()
                    if not np.isnan(mean_val):
                        result[column] = result[column].fillna(mean_val)
                    else:
                        # If mean is NaN (all values were missing), use 0
                        # This is a last resort but ensures no NaNs
                        logger.warning(
                            "Column %r has no observed values; filling with 0",
                            column,
                        )
                        result[column] = result[column].fillna(0)

        return result


class StackingImputer(BaseImputer):
    """Ensemble imputer that combines the outputs of several base imputers.

    Every base imputer is run on the same input and their completed dataframes
    are combined cell by cell. Observed values are identical in every output,
    so only the imputed cells are affected.

    **How It Works:**
    1. Run each base imputer on the dataset (failing imputers are skipped)
    2. Combine the completed dataframes with ``meta_strategy``:
       - ``"mean"``: element-wise average
       - ``"median"``: element-wise median, robust to one bad imputer
       - ``"weighted"``: reserved for learned weights; currently the same as
         ``"mean"``

    **Why Stacking Works:**
    - Reduces variance through ensemble averaging
    - Exploits diversity: Different imputers capture different patterns
    - More robust than any single method alone
    - Can outperform individual imputers, especially with complementary methods

    **Recommended Base Imputer Combinations:**
    - Simple + Complex: [MeanImputer, KNNImputer, RegressionImputer]
    - Robust mix: [MedianImputer, HuberImputer, TrimmedMeanImputer]
    - Diverse approaches: [MeanImputer, MICEImputer, MissForestImputer]

    **When to Use:**
    - When no single imputation method is clearly best
    - For production systems requiring robust performance
    - When computational cost is acceptable (runs K imputers)
    - With heterogeneous data (different columns need different methods)

    Args:
        base_imputers: List of base imputer instances to stack.
            Default: [MeanImputer(), MedianImputer()]
        meta_strategy: How to combine predictions ('mean', 'median', 'weighted').
            Default: 'mean'

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_methods import (
        ...     StackingImputer, MeanImputer, MedianImputer, KNNImputer
        ... )
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, 5]})
        >>> imputer = StackingImputer(base_imputers=[
        ...     MeanImputer(),
        ...     MedianImputer(),
        ...     KNNImputer(n_neighbors=2)
        ... ])
        >>> imputed = imputer.impute(df)

    References:
        Wolpert, D. H. (1992). Stacked generalization.
        Ensemble learning approach applied to imputation.
    """

    def __init__(
        self,
        base_imputers: list[BaseImputer] | None = None,
        meta_strategy: str = "mean",
    ) -> None:
        """Initialize the stacking imputer.

        Args:
            base_imputers: List of base imputers
            meta_strategy: Strategy for combining predictions

        Raises:
            ValueError: If meta_strategy is invalid
        """
        if meta_strategy not in ["mean", "median", "weighted"]:
            raise ValueError(
                f"meta_strategy must be 'mean', 'median', or 'weighted', "
                f"got {meta_strategy}"
            )

        if base_imputers is None:
            base_imputers = [MeanImputer(), MedianImputer()]

        self.base_imputers = base_imputers
        self.meta_strategy = meta_strategy

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using stacking ensemble.

        Executes all base imputers in parallel and combines their predictions
        using the specified meta-strategy. This approach leverages the wisdom
        of crowds - multiple diverse predictions are often better than a single
        prediction.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)

        # Phase 1: Generate predictions from all base imputers
        # Each imputer sees the same input but may use different algorithms
        # This creates diversity in the ensemble
        predictions = []
        for imputer in self.base_imputers:
            try:
                # Execute base imputer
                # Note: Each imputer handles the full dataset independently
                pred = imputer.impute(df)
                predictions.append(pred)
            except Exception as e:
                # Gracefully handle failures in individual imputers
                # Ensemble remains robust even if some members fail
                logger.warning(
                    "Base imputer %s failed (%s); excluding it",
                    type(imputer).__name__,
                    e,
                )
                continue

        # Safety check: Ensure at least one imputer succeeded
        if len(predictions) == 0:
            # All base imputers failed, fall back to simple mean
            # This ensures we always return a valid imputation
            return MeanImputer().impute(df)

        # Phase 2: Meta-learning - combine base predictions
        # This is where the "stacking" happens
        if self.meta_strategy == "mean":
            # Arithmetic mean: treats all imputers equally
            # Best when imputers have similar quality
            result = _mean_frame(predictions)

        elif self.meta_strategy == "median":
            # Element-wise median: robust to outlier predictions
            # Best when some imputers may produce bad predictions
            # More robust than mean but throws away some information
            stacked = np.stack([p.values for p in predictions], axis=0)
            result = pd.DataFrame(
                np.median(stacked, axis=0), index=df.index, columns=df.columns
            )

        else:  # weighted strategy
            # Future enhancement: learn optimal weights per imputer
            # Could use cross-validation to estimate imputer quality
            # For now, fall back to simple mean
            result = _mean_frame(predictions)

        return result


def _with_random_state(imputer: BaseImputer, seed: int) -> BaseImputer:
    """Return a copy of ``imputer`` configured with ``random_state=seed``.

    Imputers are rebuilt from their constructor arguments so that internal
    estimators created in ``__init__`` pick up the new seed. Imputers without a
    ``random_state`` parameter are deterministic and are simply copied.
    """
    parameters = inspect.signature(type(imputer).__init__).parameters
    if "random_state" not in parameters:
        return copy.deepcopy(imputer)
    kwargs: dict[str, Any] = {}
    for name, parameter in parameters.items():
        if name == "self" or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if not hasattr(imputer, name):
            clone = copy.deepcopy(imputer)
            clone.random_state = seed  # type: ignore[attr-defined]
            return clone
        kwargs[name] = copy.deepcopy(getattr(imputer, name))
    kwargs["random_state"] = seed
    return type(imputer)(**kwargs)


class BaggingImputer(BaseImputer):
    """Bootstrap aggregating (bagging) of a base imputer.

    Each of the ``n_estimators`` runs draws a bootstrap sample of the rows (with
    replacement), imputes that sample with its own copy of ``base_imputer``, and
    records the values imputed for the rows it contains. Every missing cell is
    then set to the average of the values imputed for its row across all runs
    whose sample included that row. Averaging over resampled data reduces the
    variance of unstable imputers.

    Sampled rows keep their original order, so order-dependent imputers such as
    :class:`~imputation_methods.LOCFImputer` still see a time-ordered sequence.
    Base imputers that accept ``random_state`` get a different seed for every
    run, derived from this imputer's ``random_state``. Cells whose row was never
    sampled are filled by running ``base_imputer`` once on the full data.

    **Good base imputers for bagging:** high-variance methods such as
    :class:`~imputation_methods.KNNImputer`,
    :class:`~imputation_methods.RegressionImputer` or
    :class:`~imputation_methods.PMMImputer`. Simple statistics such as the mean
    gain little.

    Args:
        base_imputer: Imputer applied to each bootstrap sample.
            Default: ``MeanImputer()``
        n_estimators: Number of bootstrap samples. Default: 10
        max_samples: Size of each bootstrap sample as a fraction of the number
            of rows. Default: 0.8
        random_state: Seed for the bootstrap samples and the base imputer
            seeds. Default: None

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from imputation_methods import BaggingImputer, KNNImputer
        >>> df = pd.DataFrame(
        ...     {
        ...         "a": [1, 2, np.nan, 4, 5, np.nan, 7],
        ...         "b": [2, 4, 6, np.nan, 10, 12, 14],
        ...     }
        ... )
        >>> imputer = BaggingImputer(
        ...     base_imputer=KNNImputer(n_neighbors=2), n_estimators=5, random_state=0
        ... )
        >>> bool(imputer.impute(df).notna().all().all())
        True

    References:
        Breiman, L. (1996). Bagging predictors. Machine Learning, 24(2), 123-140.
    """

    def __init__(
        self,
        base_imputer: BaseImputer | None = None,
        n_estimators: int = 10,
        max_samples: float = 0.8,
        random_state: int | None = None,
    ) -> None:
        """Initialize the bagging imputer.

        Args:
            base_imputer: Imputer applied to each bootstrap sample.
            n_estimators: Number of bootstrap samples.
            max_samples: Bootstrap sample size as a fraction of the rows.
            random_state: Seed for sampling and base imputer seeds.

        Raises:
            ValueError: If ``n_estimators`` or ``max_samples`` is out of range.
        """
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")
        if not 0.0 < max_samples <= 1.0:
            raise ValueError(f"max_samples must be in (0, 1], got {max_samples}")
        if base_imputer is None:
            base_imputer = MeanImputer()

        self.base_imputer = base_imputer
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using bootstrap aggregating.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe with float columns. Cells the base imputer cannot
            fill (for example in a column with no observed values) stay NaN.
        """
        df = self._ensure_numeric(df)
        values = df.to_numpy(dtype=float, na_value=np.nan)
        missing = np.isnan(values)
        if not missing.any():
            return df.astype(float)

        n_rows = len(df)
        sample_size = max(1, round(self.max_samples * n_rows))
        rng = np.random.default_rng(self.random_state)
        totals = np.zeros_like(values)
        counts = np.zeros_like(values)

        for i in range(self.n_estimators):
            rows = np.sort(rng.integers(0, n_rows, size=sample_size))
            estimator = _with_random_state(
                self.base_imputer, int(rng.integers(2**32 - 1))
            )
            sample = df.iloc[rows].reset_index(drop=True)
            try:
                imputed = estimator.impute(sample)[df.columns].to_numpy(
                    dtype=float, na_value=np.nan
                )
            except Exception as e:
                logger.warning("Estimator %d failed (%s); excluding it", i, e)
                continue
            usable = missing[rows] & ~np.isnan(imputed)
            np.add.at(totals, rows, np.where(usable, imputed, 0.0))
            np.add.at(counts, rows, usable.astype(float))

        completed = np.where(
            missing & (counts > 0), totals / np.maximum(counts, 1.0), values
        )
        uncovered = missing & (counts == 0)
        if uncovered.any():
            logger.debug(
                "%d missing cells were not in any bootstrap sample; imputing them "
                "from the full data",
                int(uncovered.sum()),
            )
            fallback = self.base_imputer.impute(df)[df.columns].to_numpy(
                dtype=float, na_value=np.nan
            )
            completed = np.where(uncovered, fallback, completed)

        return pd.DataFrame(completed, index=df.index, columns=df.columns)
