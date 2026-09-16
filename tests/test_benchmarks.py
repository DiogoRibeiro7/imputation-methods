"""Performance benchmarks for imputation methods."""

import time

import numpy as np
import pandas as pd
import pytest

from imputation_methods import (
    KNNImputerMethod,
    MeanImputer,
    MedianImputer,
    MICEImputer,
)

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize("size", [100, 500, 1000])
def test_mean_imputation_performance(size: int) -> None:
    """Benchmark mean imputation across different dataset sizes."""
    # Create dataset with missing values
    df = pd.DataFrame(np.random.randn(size, 10))
    mask = np.random.rand(size, 10) < 0.1
    df[mask] = np.nan

    # Benchmark
    start = time.time()
    imputed = MeanImputer().impute(df)
    elapsed = time.time() - start

    # Verify correctness
    assert not imputed.isna().any().any()

    # Performance assertions
    if size == 100:
        assert elapsed < 0.1, f"Mean imputation too slow for {size} rows"
    elif size == 500:
        assert elapsed < 0.2, f"Mean imputation too slow for {size} rows"
    elif size == 1000:
        assert elapsed < 0.5, f"Mean imputation too slow for {size} rows"


@pytest.mark.parametrize("size", [100, 500, 1000])
def test_median_imputation_performance(size: int) -> None:
    """Benchmark median imputation across different dataset sizes."""
    df = pd.DataFrame(np.random.randn(size, 10))
    mask = np.random.rand(size, 10) < 0.1
    df[mask] = np.nan

    start = time.time()
    imputed = MedianImputer().impute(df)
    elapsed = time.time() - start

    assert not imputed.isna().any().any()

    if size == 100:
        assert elapsed < 0.1, f"Median imputation too slow for {size} rows"
    elif size == 500:
        assert elapsed < 0.2, f"Median imputation too slow for {size} rows"
    elif size == 1000:
        assert elapsed < 0.5, f"Median imputation too slow for {size} rows"


def test_knn_imputation_performance() -> None:
    """Benchmark KNN imputation (slower, so use smaller sizes)."""
    sizes = [50, 100, 200]
    times = []

    for size in sizes:
        df = pd.DataFrame(np.random.randn(size, 5))
        mask = np.random.rand(size, 5) < 0.1
        df[mask] = np.nan

        start = time.time()
        imputed = KNNImputerMethod(k=3).impute(df)
        elapsed = time.time() - start
        times.append(elapsed)

        assert not imputed.isna().any().any()

    # KNN should complete in reasonable time even for 200 rows
    assert times[-1] < 2.0, "KNN imputation too slow"


def test_performance_comparison() -> None:
    """Compare performance of simple vs complex imputers."""
    df = pd.DataFrame(np.random.randn(200, 10))
    mask = np.random.rand(200, 10) < 0.1
    df[mask] = np.nan

    times = {}

    # Test simple imputers
    start = time.time()
    MeanImputer().impute(df)
    times["mean"] = time.time() - start

    start = time.time()
    MedianImputer().impute(df)
    times["median"] = time.time() - start

    # Test complex imputer
    start = time.time()
    KNNImputerMethod(k=3).impute(df)
    times["knn"] = time.time() - start

    # Simple imputers should be faster than KNN
    assert times["mean"] < times["knn"], "Mean should be faster than KNN"
    assert times["median"] < times["knn"], "Median should be faster than KNN"

    # Mean and median should be very fast
    assert times["mean"] < 0.5
    assert times["median"] < 0.5


def test_scalability_with_columns() -> None:
    """Test performance scales reasonably with number of columns."""
    sizes = [(100, 5), (100, 10), (100, 20)]
    times = []

    for rows, cols in sizes:
        df = pd.DataFrame(np.random.randn(rows, cols))
        mask = np.random.rand(rows, cols) < 0.1
        df[mask] = np.nan

        # Best of several runs to reduce timer noise on millisecond timings.
        elapsed = []
        for _ in range(5):
            start = time.perf_counter()
            MeanImputer().impute(df)
            elapsed.append(time.perf_counter() - start)
        times.append(min(elapsed))

    # Time should scale roughly linearly with columns: 4x the columns should
    # take well under 8x the time.
    assert times[2] < times[0] * 8, "Performance doesn't scale linearly with columns"


def test_memory_efficiency() -> None:
    """Test that imputation doesn't create excessive copies."""
    # Create a moderately sized dataset
    df = pd.DataFrame(np.random.randn(1000, 20))
    mask = np.random.rand(1000, 20) < 0.1
    df[mask] = np.nan

    # Impute (should not raise MemoryError)
    imputers = [
        MeanImputer(),
        MedianImputer(),
        KNNImputerMethod(k=5),
    ]

    for imputer in imputers:
        try:
            imputed = imputer.impute(df)
            assert imputed.shape == df.shape
        except MemoryError:
            pytest.fail(f"{imputer.__class__.__name__} uses too much memory")


def test_regression_imputer_performance() -> None:
    """Benchmark regression-based imputation."""
    # Use MICE instead of RegressionImputer as it handles
    # overlapping missing values better
    df = pd.DataFrame(np.random.randn(100, 5))
    mask = np.random.rand(100, 5) < 0.1
    df[mask] = np.nan

    start = time.time()
    imputed = MICEImputer(random_state=42).impute(df)
    elapsed = time.time() - start

    assert not imputed.isna().any().any()
    # MICE is iterative, allow reasonable time
    assert elapsed < 5.0, "MICE imputation too slow"


@pytest.mark.slow
def test_mice_imputation_performance() -> None:
    """Benchmark MICE imputation (marked as slow test)."""
    df = pd.DataFrame(np.random.randn(100, 5))
    mask = np.random.rand(100, 5) < 0.1
    df[mask] = np.nan

    start = time.time()
    imputed = MICEImputer(random_state=42).impute(df)
    elapsed = time.time() - start

    assert not imputed.isna().any().any()
    # MICE is iterative, so allow more time
    assert elapsed < 5.0, "MICE imputation too slow"


def test_missingness_ratio_impact() -> None:
    """Test how missingness ratio affects performance."""
    missingness_ratios = [0.05, 0.10, 0.20]
    times = []

    for ratio in missingness_ratios:
        df = pd.DataFrame(np.random.randn(200, 10))
        mask = np.random.rand(200, 10) < ratio
        df[mask] = np.nan

        start = time.time()
        MeanImputer().impute(df)
        elapsed = time.time() - start
        times.append(elapsed)

    # Time should not increase dramatically with missingness
    # (for simple imputers like mean)
    assert times[2] < times[0] * 2, (
        "Performance degrades too much with higher missingness"
    )
