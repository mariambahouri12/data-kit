# preprocessing/tabular/cleaners.py
import pandas as pd
import numpy as np
from typing import Optional, Any, Dict, List
from sklearn.impute import SimpleImputer, KNNImputer
from ..base import BasePreprocessor


class MissingValueCleaner(BasePreprocessor):
    """Nettoyage des valeurs manquantes"""
    
    def __init__(self, 
                 strategy: str = 'median',
                 fill_value: Optional[Any] = None,
                 columns: Optional[List[str]] = None,
                 **kwargs):
        """
        Args:
            strategy: 'mean', 'median', 'most_frequent', 'constant', 'drop', 'knn'
            fill_value: Valeur pour 'constant'
            columns: Colonnes à traiter (None = toutes)
        """
        super().__init__(**kwargs)
        # Correction: gestion de 'constant'
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns
        self.imputer = None
        self.column_types = {}
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Adapter l'imputer"""
        # Déterminer les colonnes à traiter
        if self.columns is None:
            cols_to_impute = X.columns.tolist()
        else:
            cols_to_impute = [c for c in self.columns if c in X.columns]
        
        # Séparer les types de colonnes
        numeric_cols = X[cols_to_impute].select_dtypes(include=[np.number]).columns
        categorical_cols = X[cols_to_impute].select_dtypes(include=['object', 'category']).columns
        
        self.column_types = {
            'numeric': numeric_cols.tolist(),
            'categorical': categorical_cols.tolist()
        }
        
        # Imputer pour les colonnes numériques
        if len(numeric_cols) > 0:
            if self.strategy == 'drop':
                pass
            elif self.strategy == 'knn':
                self.imputer = KNNImputer(n_neighbors=5)
                self.imputer.fit(X[numeric_cols])
            else:
                # Correction: gestion de 'constant'
                if self.strategy == 'constant' and self.fill_value is None:
                    self.fill_value = 0
                    import warnings
                    warnings.warn("fill_value is None for constant strategy, using 0 as default")
                
                self.imputer = SimpleImputer(
                    strategy=self.strategy,
                    fill_value=self.fill_value
                )
                self.imputer.fit(X[numeric_cols])
        
        # Imputer pour les colonnes catégorielles
        if len(categorical_cols) > 0:
            self.cat_imputer = SimpleImputer(
                strategy='most_frequent',
                fill_value='unknown'
            )
            self.cat_imputer.fit(X[categorical_cols])
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transformer les données"""
        X_copy = X.copy()
        
        # Traiter les colonnes numériques
        numeric_cols = self.column_types['numeric']
        if len(numeric_cols) > 0:
            if self.strategy == 'drop':
                X_copy = X_copy.dropna(subset=numeric_cols)
            elif self.imputer is not None:
                X_copy[numeric_cols] = self.imputer.transform(X_copy[numeric_cols])
        
        # Traiter les colonnes catégorielles
        categorical_cols = self.column_types['categorical']
        if len(categorical_cols) > 0 and hasattr(self, 'cat_imputer'):
            X_copy[categorical_cols] = self.cat_imputer.transform(X_copy[categorical_cols])
        
        return X_copy


class OutlierCleaner(BasePreprocessor):
    """Nettoyage des outliers"""
    
    def __init__(self, 
                 method: str = 'iqr',
                 threshold: float = 1.5,
                 action: str = 'winsorize',
                 columns: Optional[List[str]] = None,
                 **kwargs):
        """
        Args:
            method: 'iqr' ou 'zscore'
            threshold: Seuil
            action: 'winsorize' ou 'drop'
            columns: Colonnes à traiter (None = toutes)
        """
        super().__init__(**kwargs)
        self.method = method
        self.threshold = threshold
        if action == 'cap':
            action = 'winsorize'
        self.action = action
        self.columns = columns
        self.bounds = {}
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Calculer les bornes pour chaque colonne"""
        if self.columns is None:
            cols_to_clean = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols_to_clean = [c for c in self.columns if c in X.columns]
        
        for col in cols_to_clean:
            # Ignorer les NaN pour le calcul des bornes
            col_data = X[col].dropna()
            if len(col_data) == 0:
                continue
                
            if self.method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                self.bounds[col] = {
                    'lower': Q1 - self.threshold * IQR,
                    'upper': Q3 + self.threshold * IQR
                }
            elif self.method == 'zscore':
                mean = col_data.mean()
                std = col_data.std()
                self.bounds[col] = {
                    'lower': mean - self.threshold * std,
                    'upper': mean + self.threshold * std
                }
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transformer les données"""
        X_copy = X.copy()
        
        for col, bounds in self.bounds.items():
            if self.action == 'winsorize':
                X_copy[col] = X_copy[col].clip(lower=bounds['lower'], upper=bounds['upper'])
            elif self.action == 'drop':
                mask = (X_copy[col] >= bounds['lower']) & (X_copy[col] <= bounds['upper'])
                X_copy = X_copy[mask]
        
        return X_copy


class DuplicateCleaner(BasePreprocessor):
    """Nettoyage des doublons"""
    
    def __init__(self, subset: Optional[List[str]] = None, keep: str = 'first', **kwargs):
        super().__init__(**kwargs)
        self.subset = subset
        self.keep = keep
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        pass
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop_duplicates(subset=self.subset, keep=self.keep)