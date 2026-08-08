
# preprocessing/tabular/transformers/reciprocal.py

from typing import Optional, List

import numpy as np
import pandas as pd

from ...base import BasePreprocessor
from ..utils._column_utils import select_columns


class ReciprocalTransformer(BasePreprocessor):
    """Inverse transformation 1/(x + shift). Unlike Log/Sqrt, the shift
    is a fixed constant and is NOT dynamically adjusted based on the minimum
    value of the data: it should be chosen carefully if columns may contain
    values close to -shift."""

    def __init__(self, columns: Optional[List[str]] = None, shift: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.shift = shift
        self.columns_to_transform: List[str] = []

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.columns_to_transform = select_columns(
            X,
            self.columns,
            dtype_include=[np.number]
        )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        for col in self.columns_to_transform:
            X_copy[col] = 1 / (X_copy[col] + self.shift)

        return X_copy
