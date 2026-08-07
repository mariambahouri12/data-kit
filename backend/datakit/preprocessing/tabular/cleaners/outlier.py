import warnings
from typing import Optional, Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

from ...base import BasePreprocessor
from ..utils._column_utils import select_columns
from ..config import  OutlierMethod, OutlierAction


class OutlierCleaner(BasePreprocessor):
    """Détecte et traite les outliers (winsorization ou suppression)."""

    def __init__(self,
                 method: Union[str, OutlierMethod] = OutlierMethod.IQR,
                 threshold: float = 1.5,
                 action: Union[str, OutlierAction] = OutlierAction.WINSORIZE,
                 columns: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.method = OutlierMethod(method) if isinstance(method, str) else method
        self.threshold = threshold
        self.action = OutlierAction(action) if isinstance(action, str) else action
        self.columns = columns
        self.bounds: Dict[str, Dict[str, float]] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        cols_to_clean = select_columns(X, self.columns, dtype_include=[np.number])

        for col in cols_to_clean:
            col_data = X[col].dropna()
            if col_data.empty:
                continue
            self.bounds[col] = self._compute_bounds(col_data)

    def _compute_bounds(self, col_data: pd.Series) -> Dict[str, float]:
        if self.method == OutlierMethod.IQR:
            q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            iqr = q3 - q1
            return {"lower": q1 - self.threshold * iqr, "upper": q3 + self.threshold * iqr}

        if self.method == OutlierMethod.ZSCORE:
            mean, std = col_data.mean(), col_data.std()
            return {"lower": mean - self.threshold * std, "upper": mean + self.threshold * std}

        raise ValueError(f"Méthode de détection d'outliers non supportée : {self.method}")

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        for col, bounds in self.bounds.items():
            if self.action == OutlierAction.WINSORIZE:
                X_copy[col] = X_copy[col].clip(lower=bounds["lower"], upper=bounds["upper"])
            elif self.action == OutlierAction.DROP:
                mask = X_copy[col].between(bounds["lower"], bounds["upper"])
                X_copy = X_copy[mask]

        return X_copy
