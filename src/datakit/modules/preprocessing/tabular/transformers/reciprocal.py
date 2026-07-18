# preprocessing/tabular/transformers.py
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from ...base import BasePreprocessor
from .._column_utils import select_columns





class ReciprocalTransformer(BasePreprocessor):
    """Transformée inverse 1/(x + shift). Contrairement à Log/Sqrt, le shift
    est une constante fixe et n'est PAS ajustée dynamiquement au minimum des
    données : à choisir avec soin si les colonnes peuvent contenir des
    valeurs proches de -shift."""

    def __init__(self, columns: Optional[List[str]] = None, shift: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.shift = shift
        self.columns_to_transform: List[str] = []

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.columns_to_transform = select_columns(X, self.columns, dtype_include=[np.number])

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col in self.columns_to_transform:
            X_copy[col] = 1 / (X_copy[col] + self.shift)
        return X_copy