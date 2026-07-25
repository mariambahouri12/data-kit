"""
Registre central des modèles disponibles.

Contrairement à l'ancienne version qui utilisait `pkgutil` pour scanner le
package et deviner son propre chemin d'import (source de bugs fragiles liés
aux imports relatifs/absolus), les modèles s'enregistrent explicitement via
le décorateur `@register_model`, appliqué au moment de leur définition dans
chaque fichier (linear.py, tree.py, ensemble.py, knn.py). C'est explicite,
prévisible, et ne dépend d'aucune supposition sur la structure du package
appelant.
"""
import warnings
from typing import Any, Dict, List, Optional, Type

from .base import BaseModel

_MODEL_CLASSES: Dict[str, Type[BaseModel]] = {}


def register_model(cls: Type[BaseModel]) -> Type[BaseModel]:
    """Décorateur : enregistre une classe de modèle concrète sous un nom
    dérivé de son nom de classe (ex: `XGBoostModel` -> `xgboost`)."""
    name = cls.__name__.replace("Model", "").lower()
    _MODEL_CLASSES[name] = cls
    return cls


class ModelRegistry:
    """Registre singleton des modèles et de leurs schémas de paramètres."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._schemas = {}
            cls._instance._build_schemas()
        return cls._instance

    def _build_schemas(self) -> None:
        for name, model_cls in _MODEL_CLASSES.items():
            self._schemas[name] = self._get_schema(model_cls)

    @staticmethod
    def _get_schema(model_cls: Type[BaseModel]) -> Dict[str, Any]:
        try:
            temp_instance = model_cls(task="classification")
            return temp_instance.get_parameter_schema()
        except Exception as e:
            warnings.warn(f"Impossible d'obtenir le schéma pour {model_cls.__name__}: {e}")
            return {}

    def get_model(self, model_name: str, task: str = "classification", **params) -> BaseModel:
        """Instancie un modèle enregistré."""
        if model_name not in _MODEL_CLASSES:
            available = ", ".join(_MODEL_CLASSES.keys())
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")
        return _MODEL_CLASSES[model_name](task=task, **params)

    def list_models(self) -> List[Dict[str, Any]]:
        """Liste tous les modèles disponibles avec leurs métadonnées."""
        return [
            {
                "name": name,
                "class": cls.__name__,
                "available_tasks": ["classification", "regression"],
                "parameter_schema": self._schemas.get(name, {}),
            }
            for name, cls in _MODEL_CLASSES.items()
        ]

    def get_parameter_schema(self, model_name: str) -> Dict[str, Any]:
        if model_name not in _MODEL_CLASSES:
            raise ValueError(f"Model '{model_name}' not found")
        return self._schemas.get(model_name, {})

    def get_model_class(self, model_name: str) -> Optional[Type[BaseModel]]:
        return _MODEL_CLASSES.get(model_name)

    def get_available_models(self) -> List[str]:
        return list(_MODEL_CLASSES.keys())

    def clear_cache(self) -> None:
        """Recalcule les schémas (utile pour les tests)."""
        self._schemas = {}
        self._build_schemas()
