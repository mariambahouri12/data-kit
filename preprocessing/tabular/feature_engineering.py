# preprocessing/tabular/feature_engineering.py
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
import math
import warnings
from ..base import BasePreprocessor


class PolynomialFeatureCreator(BasePreprocessor):
    """Créateur de features polynomiales avec limites"""
    
    def __init__(self,
                 degree: int = 2,
                 columns: Optional[List[str]] = None,
                 interaction_only: bool = False,
                 include_bias: bool = False,
                 max_features: int = 50,
                 max_output_features: int = 5000,  # Nouveau
                 **kwargs):
        super().__init__(**kwargs)
        self.degree = degree
        self.columns = columns
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.max_features = max_features
        self.max_output_features = max_output_features
        self.poly = None
        self.feature_names = []
    
    def _calculate_poly_features(self, n: int, d: int) -> int:
        """Calculer le nombre de features polynomiales"""
        if self.interaction_only:
            # Combinaisons de 2 à d parmi n
            total = 0
            for k in range(2, d + 1):
                total += math.comb(n, k)
            return total
        else:
            # Combinaisons avec répétition
            return math.comb(n + d, d) - 1  # -1 pour exclure la constante
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.columns is None:
            cols_to_use = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols_to_use = [c for c in self.columns if c in X.columns]
        
        # Vérifier le nombre de features
        if len(cols_to_use) > self.max_features:
            raise ValueError(
                f"Too many features ({len(cols_to_use)}) for polynomial creation. "
                f"Max is {self.max_features}. Please select fewer columns or increase max_features."
            )
        
        # Vérifier le nombre de features polynomiales
        n_poly_features = self._calculate_poly_features(len(cols_to_use), self.degree)
        if n_poly_features > self.max_output_features:
            raise ValueError(
                f"Polynomial features would create {n_poly_features} features "
                f"(max is {self.max_output_features}). Please reduce degree or number of columns."
            )
        
        self.columns_to_use = cols_to_use
        
        if not cols_to_use:
            return
        
        self.poly = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        self.poly.fit(X[cols_to_use])
        self.feature_names = self.poly.get_feature_names_out(cols_to_use)
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        if not hasattr(self, 'columns_to_use') or not self.columns_to_use:
            return X_copy
        
        poly_features = self.poly.transform(X_copy[self.columns_to_use])
        poly_df = pd.DataFrame(
            poly_features,
            columns=self.feature_names,
            index=X_copy.index
        )
        
        X_copy = X_copy.drop(columns=self.columns_to_use)
        X_copy = pd.concat([X_copy, poly_df], axis=1)
        
        return X_copy


class InteractionFeatureCreator(BasePreprocessor):
    """Créateur de features d'interaction"""
    
    def __init__(self,
                 columns: Optional[List[str]] = None,
                 max_interactions: int = 2,
                 **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.max_interactions = max_interactions
        self.interaction_combinations = []
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.columns is None:
            cols_to_use = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols_to_use = [c for c in self.columns if c in X.columns]
        
        self.columns_to_use = cols_to_use
        
        self.interaction_combinations = []
        for r in range(2, min(self.max_interactions, len(cols_to_use)) + 1):
            self.interaction_combinations.extend(list(combinations(cols_to_use, r)))
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        for combo in self.interaction_combinations:
            col_name = '*'.join(combo)
            X_copy[col_name] = 1
            for col in combo:
                X_copy[col_name] = X_copy[col_name] * X_copy[col]
        
        return X_copy


class RatioFeatureCreator(BasePreprocessor):
    """Créateur de features de ratio avec limite"""
    
    def __init__(self,
                 columns: Optional[List[str]] = None,
                 epsilon: float = 1e-6,
                 max_pairs: int = 100,
                 **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.epsilon = epsilon
        self.max_pairs = max_pairs
        self.column_pairs = []
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.columns is None:
            cols_to_use = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols_to_use = [c for c in self.columns if c in X.columns]
        
        self.columns_to_use = cols_to_use
        
        # Limiter le nombre de paires
        all_pairs = list(combinations(cols_to_use, 2))
        if len(all_pairs) > self.max_pairs:
            warnings.warn(
                f"Too many pairs ({len(all_pairs)}). Limiting to {self.max_pairs}. "
                "Consider selecting fewer columns or increasing max_pairs."
            )
            all_pairs = all_pairs[:self.max_pairs]
        
        self.column_pairs = all_pairs
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        for col1, col2 in self.column_pairs:
            ratio_name = f'{col1}_over_{col2}'
            X_copy[ratio_name] = X_copy[col1] / (X_copy[col2] + self.epsilon)
            
            inverse_name = f'{col2}_over_{col1}'
            X_copy[inverse_name] = X_copy[col2] / (X_copy[col1] + self.epsilon)
        
        return X_copy


class AggregationFeatureCreator(BasePreprocessor):
    """Créateur de features par agrégation"""
    
    def __init__(self,
                 group_column: Optional[str] = None,
                 agg_columns: Optional[List[str]] = None,
                 aggregations: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.group_column = group_column
        self.agg_columns = agg_columns
        self.aggregations = aggregations or ['mean', 'sum', 'std', 'min', 'max', 'count']
        self.agg_mapping = {}
        self.agg_names = []
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.group_column is None or self.group_column not in X.columns:
            raise ValueError(f"Group column '{self.group_column}' not found")
        
        if self.agg_columns is None:
            self.agg_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            if self.group_column in self.agg_columns:
                self.agg_columns.remove(self.group_column)
        
        self.agg_mapping = {}
        
        for col in self.agg_columns:
            if col in X.columns:
                agg_dict = {}
                for agg in self.aggregations:
                    agg_dict[agg] = X.groupby(self.group_column)[col].agg(agg).to_dict()
                    self.agg_names.append(f"{col}_{agg}")
                self.agg_mapping[col] = agg_dict
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        for col, agg_dict in self.agg_mapping.items():
            for agg, mapping in agg_dict.items():
                col_name = f"{col}_{agg}"
                X_copy[col_name] = X_copy[self.group_column].map(mapping)
        
        return X_copy


class DateFeatureCreator(BasePreprocessor):
    """
    Créateur de features à partir de dates.
    ATTENTION: Seules les colonnes datetime64 sont traitées.
    """
    
    def __init__(self,
                 date_columns: Optional[List[str]] = None,
                 create_year: bool = True,
                 create_month: bool = True,
                 create_day: bool = True,
                 create_dayofweek: bool = True,
                 create_quarter: bool = True,
                 create_is_weekend: bool = True,
                 auto_detect: bool = False,  # Nouveau: désactiver par défaut
                 **kwargs):
        super().__init__(**kwargs)
        self.date_columns = date_columns
        self.create_year = create_year
        self.create_month = create_month
        self.create_day = create_day
        self.create_dayofweek = create_dayofweek
        self.create_quarter = create_quarter
        self.create_is_weekend = create_is_weekend
        self.auto_detect = auto_detect  # Désactivé par défaut pour éviter les faux positifs
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.date_columns is not None:
            cols_to_process = [c for c in self.date_columns if c in X.columns]
        elif self.auto_detect:
            # Correction: ne détecter que les colonnes datetime64
            cols_to_process = X.select_dtypes(include=['datetime64']).columns.tolist()
        else:
            cols_to_process = []
            warnings.warn(
                "No date_columns specified and auto_detect is False. "
                "Specify date_columns to process dates."
            )
        
        self.columns_to_process = cols_to_process
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        
        for col in self.columns_to_process:
            # Vérifier que c'est bien un datetime
            if not pd.api.types.is_datetime64_any_dtype(X_copy[col]):
                try:
                    X_copy[col] = pd.to_datetime(X_copy[col], errors='coerce')
                except:
                    continue
            
            # Ignorer les NaN
            if X_copy[col].isna().all():
                continue
            
            if self.create_year:
                X_copy[f'{col}_year'] = X_copy[col].dt.year
            if self.create_month:
                X_copy[f'{col}_month'] = X_copy[col].dt.month
            if self.create_day:
                X_copy[f'{col}_day'] = X_copy[col].dt.day
            if self.create_dayofweek:
                X_copy[f'{col}_dayofweek'] = X_copy[col].dt.dayofweek
            if self.create_quarter:
                X_copy[f'{col}_quarter'] = X_copy[col].dt.quarter
            if self.create_is_weekend:
                X_copy[f'{col}_is_weekend'] = (X_copy[col].dt.dayofweek >= 5).astype(int)
        
        return X_copy