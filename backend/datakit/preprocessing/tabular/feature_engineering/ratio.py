
import warnings
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ...base import BasePreprocessor
from ..utils._column_utils import select_columns


class RatioFeatureCreator(BasePreprocessor):
    """Creates ratios col_a/col_b for each pair of numerical columns."""

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        epsilon: float = 1e-6,
        max_pairs: int = 100,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.epsilon = epsilon
        self.max_pairs = max_pairs
        self.column_pairs: List[Tuple[str, str]] = []

    def _fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> None:
        cols_to_use = select_columns(
            X,
            self.columns,
            dtype_include=[np.number]
        )

        all_pairs = list(
            combinations(cols_to_use, 2)
        )

        if len(all_pairs) > self.max_pairs:
            warnings.warn(
                f"Too many pairs ({len(all_pairs)}). "
                f"Limiting to the first {self.max_pairs}. "
                "Consider selecting fewer columns or increasing max_pairs.",
                RuntimeWarning,
            )

            all_pairs = all_pairs[:self.max_pairs]

        self.column_pairs = all_pairs

    def _transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        X_copy = X.copy()

        for col1, col2 in self.column_pairs:
            X_copy[f"{col1}_over_{col2}"] = (
                X_copy[col1]
                / (X_copy[col2] + self.epsilon)
            )

            X_copy[f"{col2}_over_{col1}"] = (
                X_copy[col2]
                / (X_copy[col1] + self.epsilon)
            )

        return X_copy
