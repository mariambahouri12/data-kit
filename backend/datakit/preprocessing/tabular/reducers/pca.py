
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ...base import BasePreprocessor
from ..utils._column_utils import select_columns


class PCAReducer(BasePreprocessor):
    """Dimensionality reduction using PCA on numerical columns."""

    def __init__(
        self,
        n_components: Optional[int] = None,
        variance_ratio: float = 0.95,
        columns: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        self.columns = columns

        self.pca = None
        self.feature_names: List[str] = []
        self.columns_to_reduce: List[str] = []

    def _fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> None:
        self.columns_to_reduce = select_columns(
            X,
            self.columns,
            dtype_include=[np.number]
        )

        if not self.columns_to_reduce:
            return

        n_components = (
            self.n_components
            if self.n_components is not None
            else self.variance_ratio
        )

        self.pca = PCA(
            n_components=n_components
        )

        self.pca.fit(
            X[self.columns_to_reduce]
        )

        self.feature_names = [
            f"PC{i + 1}"
            for i in range(self.pca.n_components_)
        ]

    def _transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        if not self.columns_to_reduce:
            return X.copy()

        pca_result = self.pca.transform(
            X[self.columns_to_reduce]
        )

        pca_df = pd.DataFrame(
            pca_result,
            columns=self.feature_names,
            index=X.index
        )

        return pd.concat(
            [
                X.drop(columns=self.columns_to_reduce),
                pca_df
            ],
            axis=1
        )

    def get_explained_variance(
        self
    ) -> Dict[str, Any]:
        if self.pca is None:
            return {}

        ratios = self.pca.explained_variance_ratio_

        return {
            "explained_variance_ratio": ratios.tolist(),
            "cumulative_variance": ratios.cumsum().tolist(),
            "total_variance": float(ratios.sum()),
        }
