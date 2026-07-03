# preprocessing/tabular/transformers.py
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from sklearn.preprocessing import FunctionTransformer
from ..base import BasePreprocessor


class LogTransformer(BasePreprocessor):
    """Transformée logarithmique"""
    
    def __init__(self,
                 columns: Optional[List[str]] = None,
                 base: float = np.e,
                 shift: float = 1e-6,
                 **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.base = base
        self.shift = shift
        self.transformer = None
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.columns is None:
            cols_to_transform = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols_to_transform = [c for c in self.columns if c in X.columns]
        
        self.columns_to_transform = cols_to_transform
    
    # ✅ CORRIGÉ : Utilisation correcte de la base
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        for col in self.columns_to_transform:
            # Ajouter un shift pour éviter log(0)
            min_val = X_copy[col].min()
            shift = self.shift if min_val > 0 else abs(min_val) + self.shift
            X_copy[col] = np.log(X_copy[col] + shift) / np.log(self.base)
        
        return X_copy


class SqrtTransformer(BasePreprocessor):
    """Transformée racine carrée"""
    
    def __init__(self,
                 columns: Optional[List[str]] = None,
                 shift: float = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.shift = shift
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.columns is None:
            self.columns_to_transform = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.columns_to_transform = [c for c in self.columns if c in X.columns]
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        for col in self.columns_to_transform:
            min_val = X_copy[col].min()
            shift = self.shift if min_val >= 0 else abs(min_val) + self.shift
            X_copy[col] = np.sqrt(X_copy[col] + shift)
        
        return X_copy


class ReciprocalTransformer(BasePreprocessor):
    """Transformée inverse (1/x)"""
    
    def __init__(self,
                 columns: Optional[List[str]] = None,
                 shift: float = 1e-6,
                 **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.shift = shift
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.columns is None:
            self.columns_to_transform = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.columns_to_transform = [c for c in self.columns if c in X.columns]
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        for col in self.columns_to_transform:
            X_copy[col] = 1 / (X_copy[col] + self.shift)
        
        return X_copy


class BoxCoxTransformer(BasePreprocessor):
    """Wrapper pour Box-Cox Transformer"""
    
    def __init__(self,
                 columns: Optional[List[str]] = None,
                 lambda_: Optional[float] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.lambda_ = lambda_
        self.transformer = None
        self.lambda_estimates = {}  # ✅ INITIALISÉ
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        from scipy import stats
        
        if self.columns is None:
            cols_to_transform = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols_to_transform = [c for c in self.columns if c in X.columns]
        
        self.columns_to_transform = cols_to_transform
        
        for col in cols_to_transform:
            # ✅ Initialiser le dictionnaire pour la colonne
            self.lambda_estimates[col] = {}
            
            # Vérifier que les valeurs sont positives
            if (X[col] <= 0).any():
                # Ajouter un shift si nécessaire
                shift = abs(X[col].min()) + 1
                self.lambda_estimates[col]['shift'] = shift
                transformed = X[col] + shift
            else:
                transformed = X[col]
            
            # Estimer lambda
            if self.lambda_ is None:
                _, lambda_val = stats.boxcox(transformed)
                self.lambda_estimates[col]['lambda'] = lambda_val
            else:
                self.lambda_estimates[col]['lambda'] = self.lambda_
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        from scipy import stats
        
        X_copy = X.copy()
        
        for col in self.columns_to_transform:
            if col not in self.lambda_estimates:
                continue
                
            shift = self.lambda_estimates[col].get('shift', 0)
            lambda_val = self.lambda_estimates[col]['lambda']
            
            # Box-Cox transform
            if lambda_val == 0:
                X_copy[col] = np.log(X_copy[col] + shift)
            else:
                X_copy[col] = ((X_copy[col] + shift) ** lambda_val - 1) / lambda_val
        
        return X_copy


class YeoJohnsonTransformer(BasePreprocessor):
    """Wrapper pour Yeo-Johnson Transformer"""
    
    def __init__(self,
                 columns: Optional[List[str]] = None,
                 lambda_: Optional[float] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.lambda_ = lambda_
        self.transformer = None
        self.lambda_estimates = {}
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        from scipy import stats
        
        if self.columns is None:
            cols_to_transform = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols_to_transform = [c for c in self.columns if c in X.columns]
        
        self.columns_to_transform = cols_to_transform
        
        for col in cols_to_transform:
            self.lambda_estimates[col] = {}
            
            if self.lambda_ is None:
                _, lambda_val = stats.yeojohnson(X[col].values)
                self.lambda_estimates[col]['lambda'] = lambda_val
            else:
                self.lambda_estimates[col]['lambda'] = self.lambda_
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        from scipy import stats
        
        X_copy = X.copy()
        
        for col in self.columns_to_transform:
            if col not in self.lambda_estimates:
                continue
                
            lambda_val = self.lambda_estimates[col]['lambda']
            X_copy[col] = stats.yeojohnson(X_copy[col].values, lmbda=lambda_val)
        
        return X_copy


class PercentileTransformer(BasePreprocessor):
    """Transformée en percentiles"""
    
    def __init__(self,
                 columns: Optional[List[str]] = None,
                 n_quantiles: int = 1000,
                 **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.n_quantiles = n_quantiles
        self.quantiles = {}
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.columns is None:
            cols_to_transform = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols_to_transform = [c for c in self.columns if c in X.columns]
        
        self.columns_to_transform = cols_to_transform
        
        for col in cols_to_transform:
            quantiles = np.percentile(X[col], np.linspace(0, 100, self.n_quantiles))
            self.quantiles[col] = quantiles
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        for col in self.columns_to_transform:
            quantiles = self.quantiles[col]
            X_copy[col] = np.searchsorted(quantiles, X_copy[col]) / len(quantiles)
        
        return X_copy