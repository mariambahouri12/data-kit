# preprocessing/tabular/encoders.py
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import hashlib  # Ajout pour hash encoder reproductible
import warnings
from ..base import BasePreprocessor
from .config import EncodingMethod


class CategoricalEncoder(BasePreprocessor):
    """
    Encodeur flexible pour variables catégorielles.
    Supporte: One-Hot, Label, Target, Frequency, Binary, CatBoost, Hash, Ordinal
    """
    
    def __init__(self,
                 method: Union[str, EncodingMethod] = EncodingMethod.ONE_HOT,
                 columns: Optional[List[str]] = None,
                 max_categories: int = 50,
                 min_frequency: float = 0.01,
                 handle_unknown: str = 'ignore',
                 target: Optional[pd.Series] = None,
                 sparse: bool = True,  # Nouveau: sparse par défaut
                 **kwargs):
        """
        Args:
            method: Méthode d'encodage
            columns: Colonnes à encoder (None = toutes les catégorielles)
            max_categories: Nombre max de catégories pour One-Hot
            min_frequency: Fréquence minimum pour garder une catégorie
            handle_unknown: 'ignore' ou 'error'
            target: Target pour Target Encoding
            sparse: Utiliser sparse matrix pour One-Hot (économie de RAM)
        """
        super().__init__(**kwargs)
        self.method = EncodingMethod(method) if isinstance(method, str) else method
        self.columns = columns
        self.max_categories = max_categories
        self.min_frequency = min_frequency
        self.handle_unknown = handle_unknown
        self.target = target
        self.sparse = sparse
        self.encoder = None
        self.mapping = {}
        self.column_names = []
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Adapter l'encodeur"""
        # Déterminer les colonnes à encoder
        if self.columns is None:
            cols_to_encode = X.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            cols_to_encode = [c for c in self.columns if c in X.columns]
        
        # Filtrer les colonnes avec trop de catégories
        filtered_cols = []
        for col in cols_to_encode:
            n_unique = X[col].nunique()
            if n_unique <= self.max_categories:
                filtered_cols.append(col)
            else:
                self.mapping[col] = {'n_categories': n_unique, 'method': 'frequency'}
                warnings.warn(f"Column {col} has {n_unique} categories, using frequency encoding")
        
        self.columns_to_encode = filtered_cols
        
        if self.method == EncodingMethod.ONE_HOT:
            self._fit_onehot(X, y)
        elif self.method == EncodingMethod.LABEL:
            self._fit_label(X, y)
        elif self.method == EncodingMethod.TARGET:
            self._fit_target(X, y)
        elif self.method == EncodingMethod.FREQUENCY:
            self._fit_frequency(X, y)
        elif self.method == EncodingMethod.BINARY:
            self._fit_binary(X, y)
        elif self.method == EncodingMethod.CATBOOST:
            self._fit_catboost(X, y)
        elif self.method == EncodingMethod.HASH:
            self._fit_hash(X, y)
        elif self.method == EncodingMethod.ORDINAL:
            self._fit_ordinal(X, y)
    
    def _fit_onehot(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Adapter One-Hot Encoder avec option sparse"""
        self.encoder = OneHotEncoder(
            handle_unknown=self.handle_unknown,
            sparse_output=self.sparse,  # Correction: sparse par défaut
            drop='if_binary'
        )
        if self.columns_to_encode:
            self.encoder.fit(X[self.columns_to_encode])
            self.column_names = self.encoder.get_feature_names_out(self.columns_to_encode)
    
    def _fit_label(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Adapter Label Encoder - À utiliser avec précaution"""
        self.encoder = {}
        for col in self.columns_to_encode:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoder[col] = le
    
    def _fit_target(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Adapter Target Encoder.
        WARNING: Ceci cause du data leakage si utilisé sur tout le dataset.
        Utiliser KFold Target Encoding en production.
        """
        if y is None and self.target is None:
            raise ValueError("Target encoder requires target variable")
        
        target = y if y is not None else self.target
        
        # Correction: créer un DataFrame temporaire
        tmp = X.copy()
        tmp["_target_"] = target
        
        for col in self.columns_to_encode:
            # Calculer la moyenne de la target par catégorie
            mean_target = tmp.groupby(col)["_target_"].mean().to_dict()
            
            # Ajouter un lissage (smoothing)
            global_mean = target.mean()
            freq = tmp[col].value_counts(normalize=True).to_dict()
            
            smoothed_target = {}
            for cat, mean in mean_target.items():
                n = tmp[tmp[col] == cat].shape[0]
                smoothing = (n / (n + 10))
                smoothed_target[cat] = smoothing * mean + (1 - smoothing) * global_mean
            
            self.mapping[col] = {'encoding': smoothed_target}
    
    def _fit_frequency(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Adapter Frequency Encoder"""
        for col in self.columns_to_encode:
            freq = X[col].value_counts(normalize=True).to_dict()
            self.mapping[col] = {'encoding': freq}
    
    def _fit_binary(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Adapter Binary Encoder"""
        for col in self.columns_to_encode:
            categories = X[col].unique()
            n_bits = max(1, len(categories).bit_length())
            binary_mapping = {}
            for i, cat in enumerate(categories):
                binary_mapping[cat] = [int(b) for b in format(i, f'0{n_bits}b')]
            self.mapping[col] = {'encoding': binary_mapping, 'n_bits': n_bits}
    
    def _fit_catboost(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Adapter CatBoost Encoder"""
        if y is None and self.target is None:
            raise ValueError("CatBoost encoder requires target variable")
        
        target = y if y is not None else self.target
        
        for col in self.columns_to_encode:
            cat_means = {}
            global_mean = target.mean()
            
            for cat in X[col].unique():
                mask = X[col] == cat
                n = mask.sum()
                cat_mean = target[mask].mean() if n > 0 else global_mean
                cat_means[cat] = (n * cat_mean + global_mean) / (n + 1)
            
            self.mapping[col] = {'encoding': cat_means}
    
    def _fit_hash(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Adapter Hash Encoder avec hashlib pour reproductibilité.
        """
        for col in self.columns_to_encode:
            hash_mapping = {}
            for cat in X[col].unique():
                # Correction: utilisation de hashlib pour reproductibilité
                hash_val = int(hashlib.md5(str(cat).encode()).hexdigest(), 16) % 1000000
                hash_mapping[cat] = hash_val
            self.mapping[col] = {'encoding': hash_mapping}
    
    def _fit_ordinal(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Adapter Ordinal Encoder (plus sûr que LabelEncoder)"""
        self.encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        if self.columns_to_encode:
            self.encoder.fit(X[self.columns_to_encode])
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transformer les données"""
        X_copy = X.copy()
        
        if self.method == EncodingMethod.ONE_HOT:
            return self._transform_onehot(X_copy)
        elif self.method == EncodingMethod.ORDINAL:
            return self._transform_ordinal(X_copy)
        else:
            return self._transform_other(X_copy)
    
    def _transform_onehot(self, X: pd.DataFrame) -> pd.DataFrame:
        """Appliquer One-Hot Encoding"""
        if not self.columns_to_encode:
            return X
        
        encoded = self.encoder.transform(X[self.columns_to_encode])
        if self.sparse:
            # Convertir sparse en DataFrame
            encoded = encoded.toarray()
        
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.column_names,
            index=X.index
        )
        
        X_copy = X.drop(columns=self.columns_to_encode)
        X_copy = pd.concat([X_copy, encoded_df], axis=1)
        
        return X_copy
    
    def _transform_ordinal(self, X: pd.DataFrame) -> pd.DataFrame:
        """Appliquer Ordinal Encoding"""
        X_copy = X.copy()
        if self.columns_to_encode and self.encoder is not None:
            X_copy[self.columns_to_encode] = self.encoder.transform(X_copy[self.columns_to_encode])
        return X_copy
    
    def _transform_other(self, X: pd.DataFrame) -> pd.DataFrame:
        """Appliquer les autres encodages"""
        X_copy = X.copy()
        
        for col in self.columns_to_encode:
            if col not in X_copy.columns:
                continue
            
            if self.method == EncodingMethod.LABEL:
                le = self.encoder[col]
                # Gérer les valeurs inconnues
                try:
                    X_copy[col] = le.transform(X_copy[col].astype(str))
                except ValueError:
                    # Fallback: utiliser -1 pour les valeurs inconnues
                    known_classes = set(le.classes_)
                    X_copy[col] = X_copy[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in known_classes else -1
                    )
                
            elif self.method in [EncodingMethod.TARGET, EncodingMethod.FREQUENCY, 
                                 EncodingMethod.CATBOOST, EncodingMethod.HASH]:
                encoding_map = self.mapping[col]['encoding']
                X_copy[col] = X_copy[col].map(encoding_map)
                
                if X_copy[col].isna().any():
                    if self.handle_unknown == 'ignore':
                        X_copy[col] = X_copy[col].fillna(0)
                    else:
                        raise ValueError(f"Unknown categories found in column {col}")
                
            elif self.method == EncodingMethod.BINARY:
                n_bits = self.mapping[col]['n_bits']
                encoding_map = self.mapping[col]['encoding']
                
                for i in range(n_bits):
                    col_name = f"{col}_bit_{i}"
                    X_copy[col_name] = X_copy[col].map(
                        lambda x: encoding_map.get(x, [0]*n_bits)[i] if x in encoding_map else 0
                    )
                
                X_copy = X_copy.drop(columns=[col])
        
        return X_copy
    
    def get_feature_names(self) -> List[str]:
        """Obtenir les noms des features après encodage"""
        if self.method == EncodingMethod.ONE_HOT:
            return list(self.column_names)
        return self.columns_to_encode


class OrdinalEncoderWrapper(BasePreprocessor):
    """Wrapper pour OrdinalEncoder de sklearn avec gestion des inconnues"""
    
    def __init__(self, columns: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
    
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.columns is None:
            cols_to_encode = X.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            cols_to_encode = [c for c in self.columns if c in X.columns]
        
        self.columns_to_encode = cols_to_encode
        if cols_to_encode:
            self.encoder.fit(X[cols_to_encode])
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if self.columns_to_encode:
            X_copy[self.columns_to_encode] = self.encoder.transform(X_copy[self.columns_to_encode])
        return X_copy