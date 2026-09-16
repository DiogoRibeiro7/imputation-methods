"""Meta-imputers that combine or chain other imputers."""

from __future__ import annotations

import logging

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
    - Simple + Complex: [MeanImputer, KNNImputerMethod, RegressionImputer]
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
        ...     StackingImputer, MeanImputer, MedianImputer, KNNImputerMethod
        ... )
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, 5]})
        >>> imputer = StackingImputer(base_imputers=[
        ...     MeanImputer(),
        ...     MedianImputer(),
        ...     KNNImputerMethod(k=2)
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


class BaggingImputer(BaseImputer):
    """Average repeated runs of a base imputer.

    Runs ``base_imputer`` ``n_estimators`` times on the full dataset and
    averages the completed dataframes.

    Warning:
        Bootstrap resampling is not implemented yet: every run sees the same
        rows and uses the base imputer's own ``random_state``. Averaging only
        changes the result for base imputers that are unseeded and random, and
        ``max_samples`` is currently ignored.

    **Intended Algorithm (bootstrap aggregating):**
    1. Create n_estimators bootstrap samples (sample with replacement)
    2. Train base imputer on each bootstrap sample
    3. Predict missing values using each trained imputer
    4. Average all predictions for final result

    **Why Bagging Works:**
    - Reduces variance without increasing bias
    - Makes unstable imputers (like KNN, decision trees) more robust
    - Similar principle to Random Forests (which bags decision trees)
    - Smooths out predictions by averaging multiple noisy estimates

    **Mathematical Intuition:**
    If base imputer has variance σ², the bagged ensemble has variance ≈ σ²/n
    (assuming independent errors). More estimators = lower variance = more stable.

    **Best Base Imputers for Bagging:**
    - KNNImputerMethod (high variance method)
    - RegressionImputer (benefits from multiple training sets)
    - MissForestImputer (already uses random forests internally)
    - Avoid: MeanImputer, MedianImputer (too simple, no variance to reduce)

    **When to Use:**
    - When base imputer is unstable or high-variance
    - When you want more robust predictions
    - When computational cost is acceptable (trains n_estimators models)
    - For small to medium datasets where bootstrap sampling makes sense

    Args:
        base_imputer: Base imputer to use for each bootstrap sample.
            Default: MeanImputer()
        n_estimators: Number of bootstrap samples. Default: 10
        max_samples: Fraction of samples per bootstrap (currently unused).
            Default: 0.8
        random_state: Random seed (currently unused). Default: None

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from imputation_methods import BaggingImputer, KNNImputerMethod
        >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4, 5, np.nan, 7]})
        >>> imputer = BaggingImputer(
        ...     base_imputer=KNNImputerMethod(),
        ...     n_estimators=5
        ... )
        >>> imputed = imputer.impute(df)

    References:
        Breiman, L. (1996). Bagging predictors. Machine Learning.
        Bootstrap aggregating for variance reduction in predictions.
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
            base_imputer: Base imputer instance
            n_estimators: Number of bootstrap samples
            max_samples: Fraction of samples per bootstrap (currently unused)
            random_state: Random seed (currently unused)
        """
        if base_imputer is None:
            base_imputer = MeanImputer()

        self.base_imputer = base_imputer
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute using bagging.

        Applies the base imputer ``n_estimators`` times and averages the
        results.

        Args:
            df: Dataframe with missing values.

        Returns:
            Imputed dataframe.
        """
        df = self._ensure_numeric(df)

        all_predictions = []
        for i in range(self.n_estimators):
            try:
                all_predictions.append(self.base_imputer.impute(df))
            except Exception as e:
                logger.warning("Estimator %d failed (%s); excluding it", i, e)

        # Safety check: ensure at least one estimator succeeded
        if len(all_predictions) == 0:
            # All estimators failed, fall back to base imputer
            return self.base_imputer.impute(df)

        # Aggregate predictions via averaging
        # This is the "aggregating" part of "bootstrap aggregating"
        # Averaging reduces variance: Var(mean) = Var(X) / n
        result = _mean_frame(all_predictions)

        # Final safety check: ensure no NaNs remain
        # This should rarely trigger if base imputer is working correctly
        if result.isna().any().any():
            # Fallback imputation for any remaining NaNs
            result = result.fillna(df.mean())
            if result.isna().any().any():
                # Last resort: use zero
                result = result.fillna(0)

        return result
