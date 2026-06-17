# preprocessing/tabular/scalers.py
import pandas as pd
import numpy as np
from typing import Optional, List
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, QuantileTransformer, PowerTransformer
)
from ..base import BasePreprocessor
from .config import ScalingMethod


class FeatureScaler(BasePreprocessor):
    """Scaler flexible pour les features numériques"""
    
    def __init__(self,
                 method: str = 'standard',
                 columns: Optional[List[str]] = None,
                 with_mean: bool = True,
                 with_std: bool = True,
                 **kwargs):
        """
        Args:
            method: 'standard', 'minmax', 'robust', 'maxabs', 'quantile', 'power'
            columns: Colonnes à scaler (None = toutes les numériques)
            with_mean: StandardScaler - centrer
            with_std: StandardScaler - réduire
        """
        super().__init__(**kwargs)
        self.method = method
        self.columns = columns
        self.with_mean = with_mean
        self.with_std = with_std
        self.scaler = None
        self.scaler_type = None
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Adapter le scaler"""
        # Déterminer les colonnes
        if self.columns is None:
            cols_to_scale = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols_to_scale = [c for c in self.columns if c in X.columns]
        
        self.columns_to_scale = cols_to_scale
        
        if not cols_to_scale:
            return
        
        # Choisir le scaler
        if self.method == 'standard':
            self.scaler = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        elif self.method == 'maxabs':
            self.scaler = MaxAbsScaler()
        elif self.method == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='normal')
        elif self.method == 'power':
            self.scaler = PowerTransformer(method='yeo-johnson')
        else:
            self.scaler = StandardScaler()
        
        self.scaler.fit(X[cols_to_scale])
        self.scaler_type = type(self.scaler).__name__
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transformer les données"""
        X_copy = X.copy()
        
        if hasattr(self, 'columns_to_scale') and self.columns_to_scale:
            # Transformer et remplacer les colonnes
            scaled_data = self.scaler.transform(X_copy[self.columns_to_scale])
            X_copy[self.columns_to_scale] = scaled_data
        
        return X_copy
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverser la transformation"""
        X_copy = X.copy()
        
        if hasattr(self, 'columns_to_scale') and self.columns_to_scale:
            inv_scaled = self.scaler.inverse_transform(X_copy[self.columns_to_scale])
            X_copy[self.columns_to_scale] = inv_scaled
        
        return X_copy
    
    def get_scale_params(self) -> dict:
        """Obtenir les paramètres d'échelle"""
        if self.scaler is None:
            return {}
        
        if hasattr(self.scaler, 'scale_'):
            return {
                'mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                'scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
            }
        return {}


class PowerTransformerWrapper(BasePreprocessor):
    """Wrapper pour PowerTransformer (Box-Cox, Yeo-Johnson)"""
    
    def __init__(self,
                 method: str = 'yeo-johnson',
                 columns: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.columns = columns
        self.transformer = None
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.columns is None:
            cols_to_transform = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols_to_transform = [c for c in self.columns if c in X.columns]
        
        self.columns_to_transform = cols_to_transform
        
        if cols_to_transform:
            self.transformer = PowerTransformer(method=self.method)
            self.transformer.fit(X[cols_to_transform])
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if hasattr(self, 'columns_to_transform') and self.columns_to_transform:
            transformed = self.transformer.transform(X_copy[self.columns_to_transform])
            X_copy[self.columns_to_transform] = transformed
        return X_copy