"""Tests for the distance-based imputers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imputation_methods import (
    KNNImputerMethod,
    LocalMeanImputer,
    RadiusNeighborsImputer,
)


class TestKNNImputerMethod:
    """Tests for ``KNNImputerMethod``."""

    def test_knn_impute_basic(self) -> None:
        df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, np.nan, 7, 8]})
        imputed = KNNImputerMethod(k=2).impute(df)
        assert not imputed.isna().any().any()

    def test_knn_imputer_k_variation(self) -> None:
        """Test KNN imputer with different k values."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, 6, 7, 8]})
        imputed_k1 = KNNImputerMethod(k=1).impute(df)
        imputed_k3 = KNNImputerMethod(k=3).impute(df)
        assert not imputed_k1.isna().any().any()
        assert not imputed_k3.isna().any().any()

    def test_knn_invalid_k_type(self) -> None:
        """Test KNN imputer rejects non-integer k values."""
        with pytest.raises(TypeError, match="k must be an integer"):
            KNNImputerMethod(k=2.5)  # type: ignore

    def test_knn_invalid_k_value(self) -> None:
        """Test KNN imputer rejects non-positive k values."""
        with pytest.raises(ValueError, match="k must be positive"):
            KNNImputerMethod(k=0)
        with pytest.raises(ValueError, match="k must be positive"):
            KNNImputerMethod(k=-1)


class TestRadiusNeighborsImputer:
    """Tests for RadiusNeighborsImputer."""

    def test_basic_radius_imputation(self):
        """Test basic radius neighbors imputation."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5], "b": [2, 4, 6, np.nan, 10]})
        imputer = RadiusNeighborsImputer(radius=5.0)
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_large_radius(self):
        """Test with large radius includes all neighbors."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})
        imputer = RadiusNeighborsImputer(radius=100.0)
        result = imputer.impute(df)

        assert not result.isna().any().any()


class TestLocalMeanImputer:
    """Tests for LocalMeanImputer."""

    def test_basic_local_mean(self):
        """Test basic local weighted mean."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})
        imputer = LocalMeanImputer(n_neighbors=3)
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_distance_weight_power(self):
        """Test different distance weighting powers."""
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5]})
        imputer = LocalMeanImputer(distance_weight_power=1.0)
        result = imputer.impute(df)

        assert not result.isna().any().any()
