# preprocessing/tabular/detectors.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.feature_selection import VarianceThreshold
import warnings
from ..base import BaseDetector


class MissingValueDetector(BaseDetector):
    """Missing value detector"""
    
    def __init__(self, threshold: float = 0.05, **kwargs):
        """
        Args:
            threshold: Tolerance threshold (default: 5%)
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.missing_stats = {}
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Analyser les valeurs manquantes"""
        self.missing_stats = {
            'total_missing': X.isnull().sum().sum(),
            'total_cells': X.size,
            'missing_percentage': (X.isnull().sum().sum() / X.size) * 100,
            'columns': {}
        }
        
        self.problems = []
        
        for col in X.columns:
            missing_count = X[col].isnull().sum()
            missing_pct = (missing_count / len(X)) * 100
            
            self.missing_stats['columns'][col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_pct
            }
            
            if missing_pct > self.threshold * 100:
                severity = 'high' if missing_pct > 20 else 'medium'
                self.problems.append({
                    'column': col,
                    'description': f"{missing_pct:.1f}% de valeurs manquantes",
                    'severity': severity,
                    'suggestion': self._suggest_imputation(missing_pct)
                })
    
    def _suggest_imputation(self, missing_pct: float) -> str:
        """Suggérer une méthode d'imputation"""
        if missing_pct < 5:
            return "Supprimer les lignes ou imputer par la moyenne"
        elif missing_pct < 20:
            return "Imputer par la médiane (robuste) ou KNN"
        else:
            return "Supprimer la colonne ou imputation avancée (MICE)"
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class OutlierDetector(BaseDetector):
    """Détecteur de valeurs aberrantes"""
    
    def __init__(self, method: str = 'iqr', threshold: float = 1.5, **kwargs):
        """
        Args:
            method: 'iqr' ou 'zscore'
            threshold: Seuil (1.5 pour IQR, 3 pour z-score)
        """
        super().__init__(**kwargs)
        self.method = method
        self.threshold = threshold
        self.outlier_stats = {}
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Analyser les outliers"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.outlier_stats = {}
        self.problems = []
        
        for col in numeric_cols:
            # Ignorer les NaN pour le calcul
            col_data = X[col].dropna()
            if len(col_data) == 0:
                continue
                
            if self.method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.threshold * IQR
                upper_bound = Q3 + self.threshold * IQR
                outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)]
                n_outliers = len(outliers)
                
            elif self.method == 'zscore':
                mean = col_data.mean()
                std = col_data.std()
                if std == 0:
                    continue
                z_scores = np.abs((X[col] - mean) / std)
                outliers = X[z_scores > self.threshold]
                n_outliers = len(outliers)
            
            outlier_pct = (n_outliers / len(X)) * 100
            
            self.outlier_stats[col] = {
                'n_outliers': n_outliers,
                'percentage': outlier_pct
            }
            
            if n_outliers > 0:
                severity = 'high' if outlier_pct > 10 else 'medium'
                self.problems.append({
                    'column': col,
                    'description': f"{n_outliers} outliers ({outlier_pct:.1f}%)",
                    'severity': severity,
                    'suggestion': self._suggest_treatment(method=self.method)
                })
    
    def _suggest_treatment(self, method: str) -> str:
        """Suggérer un traitement"""
        if method == 'iqr':
            return "Winsoriser ou capper les outliers"
        return "Z-score: supprimer ou transformer"
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class CorrelationDetector(BaseDetector):
    """Détecteur de corrélations fortes"""
    
    def __init__(self, threshold: float = 0.8, **kwargs):
        """
        Args:
            threshold: Seuil de corrélation (défaut: 0.8)
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.correlations = {}
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Analyser les corrélations"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return
        
        # Correction: supprimer les NaN pour le calcul de corrélation
        data = X[numeric_cols].dropna()
        
        if len(data) < 2:
            warnings.warn("Not enough data after dropping NaNs for correlation")
            return
        
        corr_matrix = data.corr().abs()
        
        # Trouver les paires fortement corrélées
        high_corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                if corr_matrix.iloc[i, j] > self.threshold:
                    high_corr_pairs.append({
                        'col1': numeric_cols[i],
                        'col2': numeric_cols[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        self.correlations = {
            'matrix': corr_matrix,
            'high_corr_pairs': high_corr_pairs
        }
        
        if high_corr_pairs:
            for pair in high_corr_pairs:
                self.problems.append({
                    'description': f"Corrélation élevée entre {pair['col1']} et {pair['col2']}: {pair['correlation']:.2f}",
                    'severity': 'medium',
                    'suggestion': "Supprimer une des deux colonnes ou utiliser PCA"
                })
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class CardinalityDetector(BaseDetector):
    """Détecteur de cardinalité élevée"""
    
    def __init__(self, max_categories: int = 50, **kwargs):
        """
        Args:
            max_categories: Nombre maximum de catégories recommandé
        """
        super().__init__(**kwargs)
        self.max_categories = max_categories
        self.cardinality_stats = {}
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Analyser la cardinalité"""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        self.cardinality_stats = {}
        self.problems = []
        
        for col in categorical_cols:
            n_unique = X[col].nunique()
            self.cardinality_stats[col] = n_unique
            
            if n_unique > self.max_categories:
                self.problems.append({
                    'column': col,
                    'description': f"{n_unique} catégories (recommandé: < {self.max_categories})",
                    'severity': 'high' if n_unique > 100 else 'medium',
                    'suggestion': self._suggest_encoding(n_unique)
                })
    
    def _suggest_encoding(self, n_unique: int) -> str:
        """Suggérer une méthode d'encodage"""
        if n_unique < 10:
            return "One-Hot Encoding"
        elif n_unique < 50:
            return "Target Encoding"
        else:
            return "Frequency Encoding ou Binary Encoding"
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class DuplicateDetector(BaseDetector):
    """Détecteur de doublons"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.duplicate_count = 0
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Analyser les doublons"""
        self.duplicate_count = X.duplicated().sum()
        
        if self.duplicate_count > 0:
            self.problems.append({
                'description': f"{self.duplicate_count} lignes dupliquées",
                'severity': 'medium' if self.duplicate_count > 100 else 'low',
                'suggestion': "Supprimer les doublons"
            })
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X