# models/__init__.py
"""
AI Experimentation Platform - Models Package
"""

from .base import BaseModel
from .registry import ModelRegistry, register_model
from .factory import ModelFactory
from .parameter_generator import ParameterGenerator

# L'import de ces modules déclenche l'enregistrement des modèles concrets
# auprès du ModelRegistry via le décorateur @register_model (voir registry.py).
from .linear import LinearModel, RidgeModel, LassoModel, ElasticNetModel
from .tree import TreeModel, DecisionTreeModel, RandomForestModel
from .ensemble import EnsembleModel, XGBoostModel, LightGBMModel, CatBoostModel, GradientBoostingModel
from .knn import KNNModel

__all__ = [
    'BaseModel',
    'ModelRegistry',
    'register_model',
    'ModelFactory',
    'ParameterGenerator',
    'LinearModel',
    'RidgeModel',
    'LassoModel',
    'ElasticNetModel',
    'TreeModel',
    'DecisionTreeModel',
    'RandomForestModel',
    'EnsembleModel',
    'XGBoostModel',
    'LightGBMModel',
    'CatBoostModel',
    'GradientBoostingModel',
    'KNNModel',
]
