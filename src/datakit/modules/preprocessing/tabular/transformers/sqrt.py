# preprocessing/tabular/transformers.py
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from ...base import BasePreprocessor
from .._column_utils import select_columns

from ._transform_utils import _positivity_shift

class SqrtTransformer(BasePreprocessor):
    """Transformée racine carrée sqrt(x + shift). Shift calculé au fit,
    réutilisé identiquement au transform (voir LogTransformer)."""

    def __init__(self, columns: Optional[List[str]] = None, shift: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.shift = shift
        self.columns_to_transform: List[str] = []
        self.column_shifts: Dict[str, float] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.columns_to_transform = select_columns(X, self.columns, dtype_include=[np.number])
        self.column_shifts = {
            col: _positivity_shift(X[col].min(), self.shift, strict=False)
            for col in self.columns_to_transform
        }

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col in self.columns_to_transform:
            X_copy[col] = np.sqrt(X_copy[col] + self.column_shifts[col])
        return X_copy
