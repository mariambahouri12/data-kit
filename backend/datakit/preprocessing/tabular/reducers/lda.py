
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ...base import BasePreprocessor
from ..utils._column_utils import select_columns


class LDAReducer(BasePreprocessor):
    """Supervised dimensionality reduction using LDA (requires a categorical target)."""

    def __init__(
        self,
        n_components: Optional[int] = None,
        columns: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.columns = columns

        self.lda = None
        self.feature_names: List[str] = []
        self.columns_to_reduce: List[str] = []

    def _fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> None:
        if y is None:
            raise ValueError("LDA requires target variable")

        self.columns_to_reduce = select_columns(
            X,
            self.columns,
            dtype_include=[np.number]
        )

        if not self.columns_to_reduce:
            return

        n_classes = y.nunique()
        max_components = n_classes - 1

        requested = (
            self.n_components
            if self.n_components is not None
            else max_components
        )

        n_components = min(
            requested,
            max_components
        )

        if n_components < 1:
            raise ValueError(
                f"Not enough classes for LDA. "
                f"Need at least 2 classes, got {n_classes}"
            )

        self.lda = LinearDiscriminantAnalysis(
            n_components=n_components
        )

        self.lda.fit(
            X[self.columns_to_reduce],
            y
        )

        self.feature_names = [
            f"LD{i + 1}"
            for i in range(self.lda.n_components)
        ]

    def _transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        if not self.columns_to_reduce:
            return X.copy()

        lda_result = self.lda.transform(
            X[self.columns_to_reduce]
        )

        lda_df = pd.DataFrame(
            lda_result,
            columns=self.feature_names,
            index=X.index
        )

        return pd.concat(
            [
                X.drop(columns=self.columns_to_reduce),
                lda_df
            ],
            axis=1
        )

