# preprocessing/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import uuid
from datetime import datetime


class BasePreprocessor(ABC, BaseEstimator, TransformerMixin):
    """
    Classe de base pour tous les prétraitements.
    Compatible avec sklearn Pipeline.
    """
    
    def __init__(self, name: Optional[str] = None, **kwargs):
        """
        Initialiser le préprocesseur.
        
        Args:
            name: Nom du préprocesseur (auto-généré si None)
            **kwargs: Paramètres du préprocesseur
        """
        self.name = name or self.__class__.__name__
        self.id = str(uuid.uuid4())[:8]
        self.params = kwargs
        self.is_fitted = False
        self.created_at = datetime.now().isoformat()
        self._validate_params()
    
    def _validate_params(self) -> bool:
        """Valider les paramètres - à override si nécessaire"""
        return True
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Adapter le préprocesseur aux données.
        
        Args:
            X: Features
            y: Target (optionnel)
        
        Returns:
            self
        """
        self._fit(X, y)
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transformer les données.
        
        Args:
            X: Features à transformer
        
        Returns:
            Données transformées
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before transform")
        return self._transform(X)
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Adapter et transformer les données.
        
        Args:
            X: Features
            y: Target (optionnel)
        
        Returns:
            Données transformées
        """
        self.fit(X, y)
        return self.transform(X)
    
    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Implémentation de l'adaptation"""
        pass
    
    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Implémentation de la transformation"""
        pass
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Récupérer les paramètres"""
        return self.params.copy()
    
    def set_params(self, **params):
        """Définir les paramètres"""
        self.params.update(params)
        self._validate_params()
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Représentation sérialisable"""
        return {
            'id': self.id,
            'name': self.name,
            'class': self.__class__.__name__,
            'params': self.params,
            'is_fitted': self.is_fitted,
            'created_at': self.created_at
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"


class BaseDetector(BasePreprocessor):
    """
    Classe de base pour les détecteurs.
    Détecte les problèmes mais ne modifie pas les données.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.problems = []
        self.report = {}
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Les détecteurs ne transforment pas les données"""
        return X
    
    def get_report(self) -> Dict[str, Any]:
        """Récupérer le rapport de détection"""
        return {
            'problems': self.problems,
            'report': self.report,
            'summary': self._generate_summary()
        }
    
    def _generate_summary(self) -> str:
        """Générer un résumé des problèmes"""
        if not self.problems:
            return "✅ Aucun problème détecté"
        
        summary = f"⚠️ {len(self.problems)} problème(s) détecté(s):\n"
        for i, problem in enumerate(self.problems, 1):
            severity = problem.get('severity', 'info')
            emoji = '🔴' if severity == 'high' else '🟡' if severity == 'medium' else '🟢'
            summary += f"  {emoji} {problem.get('description', '')}\n"
        
        return summary


class BaseComponent(ABC):
    """
    Classe de base pour les composants du registre.
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """Obtenir le nom du composant"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Obtenir la description du composant"""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Obtenir la version du composant"""
        pass