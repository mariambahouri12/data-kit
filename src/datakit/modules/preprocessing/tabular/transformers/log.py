from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from ...base import BasePreprocessor
from .._column_utils import select_columns

from ._transform_utils import _positivity_shift

class LogTransformer(BasePreprocessor):
    """Transformée logarithmique log_base(x + shift). Le shift est calculé
    une fois au fit (à partir du minimum des données d'entraînement) puis
    réutilisé tel quel au transform, pour garantir une transformation
    identique entre train et test."""

    def __init__(self, columns: Optional[List[str]] = None, base: float = np.e,
                 shift: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.base = base
        self.shift = shift
        self.columns_to_transform: List[str] = []
        self.column_shifts: Dict[str, float] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.columns_to_transform = select_columns(X, self.columns, dtype_include=[np.number])
        self.column_shifts = {
            col: _positivity_shift(X[col].min(), self.shift, strict=True)
            for col in self.columns_to_transform
        }

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        log_base = np.log(self.base)
        for col in self.columns_to_transform:
            X_copy[col] = np.log(X_copy[col] + self.column_shifts[col]) / log_base
        return X_copy