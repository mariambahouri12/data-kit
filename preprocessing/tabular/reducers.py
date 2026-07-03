import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, f_regression,
    RFE, SelectFromModel, mutual_info_classif, mutual_info_regression
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from ..base import BasePreprocessor
from .config import TaskType


class FeatureSelector(BasePreprocessor):
    """Sélection de features par différentes méthodes"""
    
    def __init__(self,
                 method: str = 'variance',
                 threshold: float = 0.01,
                 k: Optional[int] = None,
                 columns: Optional[List[str]] = None,
                 task_type: str = 'classification',
                 **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.threshold = threshold
        self.k = k
        self.columns = columns
        self.task_type = TaskType(task_type) if isinstance(task_type, str) else task_type
        self.selector = None
        self.selected_features = []
        self.feature_importances = {}
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.columns is None:
            cols_to_select = X.columns.tolist()
        else:
            cols_to_select = [c for c in self.columns if c in X.columns]
        
        self.X_columns = cols_to_select
        X_selected = X[cols_to_select]
        
        if self.method == 'variance':
            self._fit_variance(X_selected)
        elif self.method == 'correlation':
            self._fit_correlation(X_selected, y)
        elif self.method == 'importance':
            self._fit_importance(X_selected, y)
        elif self.method == 'rfe':
            self._fit_rfe(X_selected, y)
        else:
            self.selected_features = cols_to_select
    
    def _fit_variance(self, X: pd.DataFrame):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            self.selected_features = X.columns.tolist()
            return
        
        self.selector = VarianceThreshold(threshold=self.threshold)
        self.selector.fit(X[numeric_cols])
        
        mask = self.selector.get_support()
        selected_numeric = numeric_cols[mask].tolist()
        non_numeric = [c for c in X.columns if c not in numeric_cols]
        self.selected_features = selected_numeric + non_numeric
    
    def _fit_correlation(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is None:
            self.selected_features = X.columns.tolist()
            return
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            self.selected_features = X.columns.tolist()
            return
        
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y
        
        if self.task_type == TaskType.CLASSIFICATION:
            scores = mutual_info_classif(X[numeric_cols], y_encoded, random_state=42)
        else:
            scores = mutual_info_regression(X[numeric_cols], y_encoded, random_state=42)
        
        sorted_cols = sorted(
            zip(numeric_cols, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        if self.k:
            selected = [col for col, _ in sorted_cols[:self.k]]
        else:
            selected = [col for col, score in sorted_cols if score >= self.threshold]
        
        non_numeric = [c for c in X.columns if c not in numeric_cols]
        self.selected_features = selected + non_numeric
        self.feature_importances = dict(sorted_cols)
    
    def _fit_importance(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is None:
            self.selected_features = X.columns.tolist()
            return
        
        if self.task_type == TaskType.CLASSIFICATION:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
            X_encoded[col] = X_encoded[col].astype('category').cat.codes
        
        model.fit(X_encoded, y)
        
        importances = model.feature_importances_
        feature_names = X_encoded.columns
        
        importance_dict = dict(zip(feature_names, importances))
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        if self.k:
            selected = [col for col, _ in sorted_features[:self.k]]
        else:
            selected = [col for col, imp in sorted_features if imp >= self.threshold]
        
        self.selected_features = selected
        self.feature_importances = dict(sorted_features)
    
    def _fit_rfe(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is None:
            self.selected_features = X.columns.tolist()
            return
        
        if self.task_type == TaskType.CLASSIFICATION:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        n_features = self.k if self.k else max(1, len(X.columns) // 2)
        
        self.selector = RFE(model, n_features_to_select=n_features)
        self.selector.fit(X, y)
        
        mask = self.selector.get_support()
        self.selected_features = X.columns[mask].tolist()
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features:
            return X
        return X[self.selected_features]


class PCAReducer(BasePreprocessor):
    """Réduction par PCA"""
    
    def __init__(self,
                 n_components: Optional[int] = None,
                 variance_ratio: float = 0.95,
                 columns: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        self.columns = columns
        self.pca = None
        self.feature_names = []
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.columns is None:
            cols_to_reduce = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols_to_reduce = [c for c in self.columns if c in X.columns]
        
        self.columns_to_reduce = cols_to_reduce
        
        if not cols_to_reduce:
            return
        
        if self.n_components:
            self.pca = PCA(n_components=self.n_components)
        else:
            self.pca = PCA(n_components=self.variance_ratio)
        
        self.pca.fit(X[cols_to_reduce])
        
        n_components = self.pca.n_components_
        self.feature_names = [f'PC{i+1}' for i in range(n_components)]
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        if not hasattr(self, 'columns_to_reduce') or not self.columns_to_reduce:
            return X_copy
        
        pca_result = self.pca.transform(X_copy[self.columns_to_reduce])
        pca_df = pd.DataFrame(pca_result, columns=self.feature_names, index=X_copy.index)
        
        X_copy = X_copy.drop(columns=self.columns_to_reduce)
        X_copy = pd.concat([X_copy, pca_df], axis=1)
        
        return X_copy
    
    def get_explained_variance(self) -> dict:
        if self.pca is None:
            return {}
        
        return {
            'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': self.pca.explained_variance_ratio_.cumsum().tolist(),
            'total_variance': self.pca.explained_variance_ratio_.sum()
        }


class LDAReducer(BasePreprocessor):
    """Réduction par LDA (Linear Discriminant Analysis)"""
    
    def __init__(self,
                 n_components: Optional[int] = None,
                 columns: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.columns = columns
        self.lda = None
        self.feature_names = []
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is None:
            raise ValueError("LDA requires target variable")
        
        if self.columns is None:
            cols_to_reduce = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols_to_reduce = [c for c in self.columns if c in X.columns]
        
        self.columns_to_reduce = cols_to_reduce
        
        if not cols_to_reduce:
            return
        
        n_classes = y.nunique()
        n_components = min(self.n_components or n_classes - 1, n_classes - 1)
        
        if n_components < 1:
            raise ValueError(f"Not enough classes for LDA. Need at least 2 classes, got {n_classes}")
        
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        self.lda.fit(X[cols_to_reduce], y)
        
        # Correction : LDA n'a pas n_components_, on utilise n_components
        n_components_actual = self.lda.n_components
        self.feature_names = [f'LD{i+1}' for i in range(n_components_actual)]
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        if not hasattr(self, 'columns_to_reduce') or not self.columns_to_reduce:
            return X_copy
        
        lda_result = self.lda.transform(X_copy[self.columns_to_reduce])
        lda_df = pd.DataFrame(lda_result, columns=self.feature_names, index=X_copy.index)
        
        X_copy = X_copy.drop(columns=self.columns_to_reduce)
        X_copy = pd.concat([X_copy, lda_df], axis=1)
        
        return X_copy