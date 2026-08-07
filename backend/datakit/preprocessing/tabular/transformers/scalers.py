# preprocessing/tabular/scalers.py
from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, QuantileTransformer, PowerTransformer,
)

from ...base import BasePreprocessor
from ..utils._column_utils import select_columns
from ..config import ScalingMethod


class FeatureScaler(BasePreprocessor):
    """Scaler flexible pour les features numériques.

    Supporte : standard, minmax, robust, maxabs, quantile, power (Yeo-Johnson
    ou Box-Cox selon `power_method`).
    """

    def __init__(self,
                 method: Union[str, ScalingMethod] = ScalingMethod.STANDARD,
                 columns: Optional[List[str]] = None,
                 with_mean: bool = True,
                 with_std: bool = True,
                 power_method: str = "yeo-johnson",
                 **kwargs):
        """
        Args:
            method: Méthode de mise à l'échelle.
            columns: Colonnes à scaler (None = toutes les numériques).
            with_mean: StandardScaler - centrer.
            with_std: StandardScaler - réduire.
            power_method: 'yeo-johnson' (gère les valeurs négatives) ou
                'box-cox' (valeurs strictement positives uniquement),
                utilisé seulement si method == ScalingMethod.POWER.
        """
        super().__init__(**kwargs)
        self.method = ScalingMethod(method) if isinstance(method, str) else method
        self.columns = columns
        self.with_mean = with_mean
        self.with_std = with_std
        self.power_method = power_method

        self.scaler = None
        self.scaler_type: Optional[str] = None
        self.columns_to_scale: List[str] = []

    _SIMPLE_SCALERS = {
        ScalingMethod.MINMAX: MinMaxScaler,
        ScalingMethod.ROBUST: RobustScaler,
        ScalingMethod.MAXABS: MaxAbsScaler,
    }

    def _build_scaler(self):
        if self.method == ScalingMethod.STANDARD:
            return StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        if self.method == ScalingMethod.QUANTILE:
            return QuantileTransformer(output_distribution="normal")
        if self.method == ScalingMethod.POWER:
            return PowerTransformer(method=self.power_method)
        if self.method in self._SIMPLE_SCALERS:
            return self._SIMPLE_SCALERS[self.method]()
        raise ValueError(f"Méthode de scaling non supportée : {self.method}")

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.columns_to_scale = select_columns(X, self.columns, dtype_include=[np.number])
        if not self.columns_to_scale:
            return

        self.scaler = self._build_scaler()
        self.scaler.fit(X[self.columns_to_scale])
        self.scaler_type = type(self.scaler).__name__

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if self.columns_to_scale:
            X_copy[self.columns_to_scale] = self.scaler.transform(X_copy[self.columns_to_scale])
        return X_copy

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse la transformation (non supporté par QuantileTransformer avec certaines
        configurations ni par PowerTransformer sur des colonnes hors du domaine d'origine)."""
        X_copy = X.copy()
        if self.columns_to_scale:
            X_copy[self.columns_to_scale] = self.scaler.inverse_transform(X_copy[self.columns_to_scale])
        return X_copy

    def get_scale_params(self) -> Dict[str, Any]:
        """Paramètres d'échelle appris. Le contenu dépend du type de scaler
        (ex: mean_/scale_ pour StandardScaler, data_min_/data_max_ pour MinMaxScaler)."""
        if self.scaler is None:
            return {}

        params = {}
        for attr in ("mean_", "scale_", "center_", "data_min_", "data_max_"):
            if hasattr(self.scaler, attr):
                params[attr.rstrip("_")] = getattr(self.scaler, attr).tolist()

        return params