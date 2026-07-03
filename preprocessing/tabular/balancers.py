# preprocessing/tabular/balancers.py
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.preprocessing import LabelEncoder
from ..base import BasePreprocessor
from .config import BalancingMethod


class ClassBalancer(BasePreprocessor):
    """
    Balancer flexible pour classes déséquilibrées.
    Supporte: SMOTE, ADASYN, Random Over/Under, Tomek, ENN
    """
    
    def __init__(self,
                 method: Union[str, BalancingMethod] = BalancingMethod.SMOTE,
                 target_column: Optional[str] = None,
                 sampling_strategy: Optional[Dict] = None,
                 random_state: int = 42,
                 **kwargs):
        """
        Args:
            method: Méthode de rééquilibrage
            target_column: Colonne cible (si non fournie, sera détectée)
            sampling_strategy: Stratégie d'échantillonnage
            random_state: Seed aléatoire
        """
        super().__init__(**kwargs)
        self.method = BalancingMethod(method) if isinstance(method, str) else method
        self.target_column = target_column
        self.sampling_strategy = sampling_strategy or 'auto'
        self.random_state = random_state
        self.balancer = None
        self.encoder = None
        self.original_shape = None
        self.balanced_shape = None
        self.target = None
    
    # ✅ AJOUT : Méthode _fit (abstraite)
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Adapter le balancer.
        Note: Le fit est différent car il doit retourner les données rééquilibrées.
        """
        self.original_shape = X.shape
        
        # Pour les méthodes qui ne supportent pas le fit_transform
        if self.method == BalancingMethod.NONE:
            return self
        
        self.is_fitted = True
        return self
    
    # ✅ AJOUT : Méthode _transform (abstraite)
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transformer les données.
        Note: Retourne uniquement X, y est géré séparément.
        """
        if not self.is_fitted:
            raise ValueError("Balancer must be fitted first")
        
        return X
    
    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Adapter et rééquilibrer les données.
        
        Args:
            X: Features
            y: Target
        
        Returns:
            (X_resampled, y_resampled)
        """
        self.original_shape = X.shape
        self.target = y
        
        # Encoder la target si nécessaire
        if y.dtype == 'object' or y.dtype.name == 'category':
            self.encoder = LabelEncoder()
            y_encoded = self.encoder.fit_transform(y)
        else:
            y_encoded = y
        
        # Choisir le balancer
        if self.method == BalancingMethod.SMOTE:
            self.balancer = SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
        elif self.method == BalancingMethod.ADASYN:
            self.balancer = ADASYN(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
        elif self.method == BalancingMethod.RANDOM_OVER:
            self.balancer = RandomOverSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
        elif self.method == BalancingMethod.RANDOM_UNDER:
            self.balancer = RandomUnderSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
        elif self.method == BalancingMethod.TOMEK:
            self.balancer = TomekLinks(sampling_strategy=self.sampling_strategy)
        elif self.method == BalancingMethod.ENN:
            self.balancer = EditedNearestNeighbours(sampling_strategy=self.sampling_strategy)
        elif self.method == BalancingMethod.NONE:
            return X, y
        else:
            # Combinaison
            self.balancer = SMOTETomek(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
        
        # Appliquer le rééquilibrage
        try:
            X_resampled, y_resampled = self.balancer.fit_resample(X, y_encoded)
        except Exception as e:
            print(f"⚠️ Warning: {e}. Using Random Over-sampling as fallback.")
            fallback = RandomOverSampler(random_state=self.random_state)
            X_resampled, y_resampled = fallback.fit_resample(X, y_encoded)
        
        # Décoder la target si nécessaire
        if self.encoder is not None:
            y_resampled = self.encoder.inverse_transform(y_resampled)
        
        self.balanced_shape = X_resampled.shape
        self.is_fitted = True
        
        return X_resampled, y_resampled
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Adapter et transformer (alias pour fit_resample)"""
        return self.fit_resample(X, y)
    
    def get_balance_report(self) -> Dict[str, Any]:
        """Obtenir un rapport sur le rééquilibrage"""
        if self.original_shape is None:
            return {}
        
        return {
            'method': self.method.value if hasattr(self.method, 'value') else str(self.method),
            'original_shape': self.original_shape,
            'balanced_shape': self.balanced_shape,
            'sampling_strategy': self.sampling_strategy,
            'random_state': self.random_state
        }
    
    def get_class_distribution(self, y: pd.Series) -> Dict[str, Any]:
        """Obtenir la distribution des classes"""
        value_counts = y.value_counts()
        n_classes = len(value_counts)
        min_class = value_counts.min()
        max_class = value_counts.max()
        
        return {
            'n_classes': n_classes,
            'counts': value_counts.to_dict(),
            'percentages': (value_counts / len(y) * 100).to_dict(),
            'imbalance_ratio': max_class / min_class if min_class > 0 else float('inf')
        }
    
    def suggest_balancing_method(self, y: pd.Series) -> Dict[str, Any]:
        """
        Suggérer une méthode de rééquilibrage basée sur les données.
        """
        distribution = self.get_class_distribution(y)
        imbalance_ratio = distribution['imbalance_ratio']
        
        suggestions = []
        
        if imbalance_ratio < 2:
            suggestions.append({
                'method': BalancingMethod.NONE,
                'reason': "Les classes sont déjà équilibrées",
                'priority': 1
            })
        elif imbalance_ratio < 5:
            suggestions.append({
                'method': BalancingMethod.RANDOM_OVER,
                'reason': "Déséquilibre modéré, over-sampling simple",
                'priority': 1
            })
            suggestions.append({
                'method': BalancingMethod.SMOTE,
                'reason': "Déséquilibre modéré, SMOTE donne de meilleurs résultats",
                'priority': 2
            })
        elif imbalance_ratio < 10:
            suggestions.append({
                'method': BalancingMethod.SMOTE,
                'reason': "Déséquilibre important, SMOTE est recommandé",
                'priority': 1
            })
            suggestions.append({
                'method': BalancingMethod.ADASYN,
                'reason': "Alternative à SMOTE pour les cas difficiles",
                'priority': 2
            })
        else:
            suggestions.append({
                'method': BalancingMethod.ADASYN,
                'reason': "Déséquilibre sévère, ADASYN est recommandé",
                'priority': 1
            })
            suggestions.append({
                'method': BalancingMethod.SMOTE,
                'reason': "Alternative pour déséquilibre sévère",
                'priority': 2
            })
        
        return {
            'imbalance_ratio': imbalance_ratio,
            'severity': 'low' if imbalance_ratio < 2 else 'medium' if imbalance_ratio < 5 else 'high',
            'suggestions': suggestions
        }