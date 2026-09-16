"""Tests for the iterative multivariate imputers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from imputation_methods import (
    EMImputer,
    MICEImputer,
    MissForestImputer,
)


class TestMICEImputer:
    """Tests for ``MICEImputer``."""

    def test_mice_impute_basic(self) -> None:
        df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, np.nan, 7, 8]})
        imputed = MICEImputer(random_state=0).impute(df)
        assert not imputed.isna().any().any()


class TestEMImputer:
    """Tests for EMImputer."""

    def test_basic_imputation(self):
        """Test basic EM imputation."""
        df = pd.DataFrame(
            {"a": [1.0, 2.0, np.nan, 4.0, 5.0], "b": [5.0, np.nan, 7.0, 8.0, 9.0]}
        )
        imputer = EMImputer(max_iter=50)
        result = imputer.impute(df)

        assert not result.isna().any().any()
        assert result.shape == df.shape

    def test_convergence_parameters(self):
        """Test that convergence parameters are respected."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [5.0, np.nan, 7.0, 8.0]})
        imputer = EMImputer(max_iter=10, tol=1e-3)
        result = imputer.impute(df)

        assert not result.isna().any().any()

    def test_reproducibility(self):
        """Test that results are reproducible with random_state."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})

        imputer1 = EMImputer(random_state=42)
        result1 = imputer1.impute(df)

        imputer2 = EMImputer(random_state=42)
        result2 = imputer2.impute(df)

        pd.testing.assert_frame_equal(result1, result2)


class TestMissForestImputer:
    """Tests for ``MissForestImputer``."""

    def test_miss_forest_imputer(self) -> None:
        df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [5, 6, 7, np.nan]})
        imputed = MissForestImputer(random_state=0).impute(df)
        assert not imputed.isna().any().any()
