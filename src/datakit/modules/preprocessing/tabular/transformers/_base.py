# preprocessing/tabular/transformers.py
from typing import Optional, List, Dict

import numpy as np
import pandas as pd


from ...base import BasePreprocessor
from .._column_utils import select_columns


class _LambdaFamilyTransformer(BasePreprocessor):
    """Base commune pour les transformées de la famille 'puissance' (Box-Cox,
    Yeo-Johnson), qui estiment un paramètre lambda par colonne au fit et
    l'appliquent tel quel au transform. Les sous-classes définissent
    seulement l'estimation et la formule de transformation par colonne."""

    def __init__(self, columns: Optional[List[str]] = None,
                 lambda_: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.lambda_ = lambda_
        self.columns_to_transform: List[str] = []
        self.column_params: Dict[str, Dict[str, float]] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.columns_to_transform = select_columns(X, self.columns, dtype_include=[np.number])
        self.column_params = {
            col: self._estimate_column(X[col]) for col in self.columns_to_transform
        }

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col in self.columns_to_transform:
            params = self.column_params.get(col)
            if params is None:
                continue
            X_copy[col] = self._transform_column(X_copy[col], params)
        return X_copy

    def _estimate_column(self, series: pd.Series) -> Dict[str, float]:
        raise NotImplementedError

    def _transform_column(self, series: pd.Series, params: Dict[str, float]) -> pd.Series:
        raise NotImplementedError
