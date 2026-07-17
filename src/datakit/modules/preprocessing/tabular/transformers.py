# preprocessing/tabular/transformers.py
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from ..base import BasePreprocessor
from ._column_utils import select_columns


def _positivity_shift(min_val: float, base_shift: float, strict: bool) -> float:
    """Décalage additif pour garantir que toutes les valeurs deviennent
    strictement positives (strict=True) ou positives ou nulles (strict=False)."""
    if strict:
        return base_shift if min_val > 0 else abs(min_val) + base_shift
    return base_shift if min_val >= 0 else abs(min_val) + base_shift


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


class BoxCoxTransformer(_LambdaFamilyTransformer):
    """Transformée Box-Cox. Nécessite des valeurs strictement positives :
    un shift est ajouté automatiquement si la colonne contient des valeurs <= 0."""

    def _estimate_column(self, series: pd.Series) -> Dict[str, float]:
        params: Dict[str, float] = {}
        shifted = series

        if (series <= 0).any():
            shift = abs(series.min()) + 1
            params["shift"] = shift
            shifted = series + shift

        params["lambda"] = self.lambda_ if self.lambda_ is not None else stats.boxcox(shifted)[1]
        return params

    def _transform_column(self, series: pd.Series, params: Dict[str, float]) -> pd.Series:
        shift = params.get("shift", 0)
        lambda_val = params["lambda"]
        shifted = series + shift

        if lambda_val == 0:
            return np.log(shifted)
        return (shifted ** lambda_val - 1) / lambda_val


class YeoJohnsonTransformer(_LambdaFamilyTransformer):
    """Transformée Yeo-Johnson : comme Box-Cox mais accepte les valeurs
    négatives, donc aucun shift n'est nécessaire."""

    def _estimate_column(self, series: pd.Series) -> Dict[str, float]:
        lambda_val = self.lambda_ if self.lambda_ is not None else stats.yeojohnson(series.values)[1]
        return {"lambda": lambda_val}

    def _transform_column(self, series: pd.Series, params: Dict[str, float]) -> pd.Series:
        return stats.yeojohnson(series.values, lmbda=params["lambda"])


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