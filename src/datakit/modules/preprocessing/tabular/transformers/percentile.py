# preprocessing/tabular/transformers.py
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from ...base import BasePreprocessor
from .._column_utils import select_columns

class PercentileTransformer(BasePreprocessor):
    """Transforme chaque valeur en son rang percentile (0 à 1) par rapport
    à la distribution apprise au fit.

    Note : recoupe fonctionnellement FeatureScaler(method=ScalingMethod.QUANTILE)
    dans scalers.py, qui s'appuie sur sklearn.QuantileTransformer (plus robuste,
    gère nativement les valeurs hors-plage). À utiliser seulement si le rang
    percentile brut (plutôt qu'une distribution normale/uniforme) est explicitement
    voulu ; sinon préférer FeatureScaler.
    """

    def __init__(self, columns: Optional[List[str]] = None, n_quantiles: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.n_quantiles = n_quantiles
        self.columns_to_transform: List[str] = []
        self.quantiles: Dict[str, np.ndarray] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.columns_to_transform = select_columns(X, self.columns, dtype_include=[np.number])
        self.quantiles = {
            col: np.percentile(X[col], np.linspace(0, 100, self.n_quantiles))
            for col in self.columns_to_transform
        }

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col in self.columns_to_transform:
            quantiles = self.quantiles[col]
            X_copy[col] = np.searchsorted(quantiles, X_copy[col]) / len(quantiles)
        return X_copy