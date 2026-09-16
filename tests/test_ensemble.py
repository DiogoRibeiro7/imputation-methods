"""Tests for the ensemble meta-imputers."""

from __future__ import annotations

import inspect
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

import imputation_methods
from imputation_methods import (
    BaggingImputer,
    BaseImputer,
    HybridImputer,
    InterpolationImputer,
    KNNImputer,
    MeanImputer,
    MedianImputer,
    PMMImputer,
    RandomSamplingImputer,
    StackingImputer,
    bagging_impute,
)
from imputation_methods.ensemble import _with_random_state


class RecordingImputer(BaseImputer):
    """Mean imputer that records what each bagging run receives."""

    calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        RecordingImputer.calls.append(
            {"row_ids": df["row_id"].tolist(), "seed": self.random_state}
        )
        return df.fillna(df.mean())


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame(
        {
            "row_id": np.arange(n, dtype=float),
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
        }
    )
    df.loc[rng.choice(n, 10, replace=False), "a"] = np.nan
    return df


def _seeded_imputer_classes() -> list[type[BaseImputer]]:
    return [
        obj
        for obj in vars(imputation_methods).values()
        if inspect.isclass(obj)
        and issubclass(obj, BaseImputer)
        and obj is not BaseImputer
        and "random_state" in inspect.signature(obj).parameters
    ]


class TestHybridImputer:
    """Tests for HybridImputer."""

    def test_hybrid_default(self):
        """Test hybrid imputer with default methods."""
        df = pd.DataFrame({"a": [1, np.nan, 3, np.nan, 5]})
        imputer = HybridImputer()
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_hybrid_custom_chain(self):
        """Test hybrid with custom method chain."""
        df = pd.DataFrame({"a": [1, np.nan, np.nan, 4, np.nan]})
        imputer = HybridImputer(methods=[InterpolationImputer(), MeanImputer()])
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_hybrid_fallback_chain(self):
        """Test that hybrid tries multiple methods."""
        df = pd.DataFrame({"a": [np.nan, np.nan, 3, 4, 5]})
        # Interpolation might struggle with leading NaNs, should fall back
        imputer = HybridImputer(methods=[InterpolationImputer(), MeanImputer()])
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_hybrid_all_nan_fallback(self):
        """Test hybrid final fallback when all values missing."""
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        imputer = HybridImputer()
        result = imputer.impute(df)

        # Should fall back to 0
        assert all(result["a"] == 0)


class TestStackingImputer:
    """Tests for StackingImputer."""

    def test_default_stacking(self):
        """Test stacking with default imputers."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})
        imputer = StackingImputer()
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_custom_base_imputers(self):
        """Test with custom base imputers."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})
        imputer = StackingImputer(base_imputers=[MeanImputer(), MedianImputer()])
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_meta_strategy_median(self):
        """Test median meta strategy."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})
        imputer = StackingImputer(meta_strategy="median")
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_invalid_meta_strategy(self):
        """Test invalid meta strategy raises error."""
        with pytest.raises(ValueError, match="meta_strategy must be"):
            StackingImputer(meta_strategy="invalid")


class TestBaggingImputer:
    """Tests for BaggingImputer."""

    def test_basic_bagging(self):
        """Test basic bagging imputation."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5, np.nan, 7]})
        imputer = BaggingImputer(n_estimators=5, random_state=42)
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_with_custom_base_imputer(self):
        """Test bagging with custom base imputer."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})
        imputer = BaggingImputer(base_imputer=MedianImputer(), n_estimators=3)
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_each_run_gets_a_sorted_bootstrap_sample_and_its_own_seed(
        self,
        frame: pd.DataFrame,
    ) -> None:
        RecordingImputer.calls = []
        BaggingImputer(
            base_imputer=RecordingImputer(),
            n_estimators=4,
            max_samples=0.8,
            random_state=0,
        ).impute(frame)

        runs = RecordingImputer.calls[:4]
        assert len(runs) >= 4
        for run in runs:
            row_ids = run["row_ids"]
            assert len(row_ids) == 40
            assert row_ids == sorted(row_ids)
        # Sampling is with replacement, so duplicates appear across the runs.
        assert any(len(set(run["row_ids"])) < len(run["row_ids"]) for run in runs)
        seeds = [run["seed"] for run in runs]
        assert len(set(seeds)) == len(seeds)

    def test_is_reproducible_with_random_state(self, frame: pd.DataFrame) -> None:
        base = PMMImputer(n_neighbors=3, random_state=0)
        first = BaggingImputer(
            base_imputer=base, n_estimators=5, random_state=1
        ).impute(frame)
        second = bagging_impute(
            frame, base_imputer=base, n_estimators=5, random_state=1
        )
        pd.testing.assert_frame_equal(first, second)

    def test_averages_differ_from_single_runs(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0]})
        result = BaggingImputer(
            RandomSamplingImputer(), n_estimators=10, max_samples=1.0, random_state=0
        ).impute(df)
        imputed = result.loc[[2, 6], "a"]
        # A single random-sampling run can only return observed values; the bagged
        # value is an average of many draws.
        assert not set(imputed) <= set(df["a"].dropna())
        assert imputed.between(1.0, 8.0).all()

    def test_reduces_error_of_a_high_variance_imputer(self) -> None:
        rng = np.random.default_rng(5)
        values = rng.normal(size=(300, 2)) @ rng.normal(size=(2, 6))
        complete = pd.DataFrame(values + 0.1 * rng.normal(size=values.shape))
        mask = rng.random(complete.shape) < 0.2
        missing = complete.mask(mask)

        def rmse(imputed: pd.DataFrame) -> float:
            diff = imputed.to_numpy()[mask] - complete.to_numpy()[mask]
            return float(np.sqrt(np.mean(diff**2)))

        base = PMMImputer(n_neighbors=3, random_state=0)
        bagged = BaggingImputer(
            base_imputer=base, n_estimators=20, max_samples=1.0, random_state=0
        )
        assert rmse(bagged.impute(missing)) < 0.9 * rmse(base.impute(missing))

    def test_rows_outside_every_sample_use_the_full_data_fallback(
        self,
        frame: pd.DataFrame,
    ) -> None:
        result = BaggingImputer(
            base_imputer=MeanImputer(), n_estimators=1, max_samples=0.1, random_state=0
        ).impute(frame)
        assert result.notna().all().all()

    def test_empty_column_stays_missing(self, frame: pd.DataFrame) -> None:
        result = BaggingImputer(n_estimators=3, random_state=0).impute(
            frame.assign(empty=np.nan)
        )
        assert result["empty"].isna().all()
        assert result[["a", "b"]].notna().all().all()

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n_estimators": 0}, "n_estimators"),
            ({"max_samples": 0.0}, "max_samples"),
            ({"max_samples": 1.5}, "max_samples"),
        ],
    )
    def test_validates_arguments(self, kwargs: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            BaggingImputer(**kwargs)


class TestWithRandomState:
    """Tests for the helper that gives each bagging run its own seed."""

    @pytest.mark.parametrize(
        "cls", _seeded_imputer_classes(), ids=lambda cls: cls.__name__
    )
    def test_with_random_state_rebuilds_every_seeded_imputer(
        self,
        cls: type[BaseImputer],
    ) -> None:
        kwargs = {"group_col": "g"} if cls.__name__ == "GroupMeanImputer" else {}
        original = cls(**kwargs)

        clone = _with_random_state(original, 1234)

        assert type(clone) is cls
        assert clone is not original
        assert clone.random_state == 1234  # type: ignore[attr-defined]
        assert original.random_state is None  # type: ignore[attr-defined]

    def test_with_random_state_copies_deterministic_imputers(self) -> None:
        original = KNNImputer(n_neighbors=3)
        clone = _with_random_state(original, 7)
        assert clone is not original
        assert clone.n_neighbors == 3  # type: ignore[attr-defined]
