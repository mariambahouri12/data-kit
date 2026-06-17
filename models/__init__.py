# models/__init__.py
"""
AI Experimentation Platform - Models Package
"""

from .base import BaseModel
from .registry import ModelRegistry
from .factory import ModelFactory
from .parameter_generator import ParameterGenerator

# Import models to register them
from .linear import LinearModel, RidgeModel, LassoModel, ElasticNetModel
from .tree import TreeModel, RandomForestModel
from .ensemble import XGBoostModel, LightGBMModel, CatBoostModel, GradientBoostingModel
from .knn import KNNModel

__all__ = [
    'BaseModel',
    'ModelRegistry',
    'ModelFactory',
    'ParameterGenerator',
    'LinearModel',
    'RidgeModel',
    'LassoModel',
    'ElasticNetModel',
    'TreeModel',
    'RandomForestModel',
    'XGBoostModel',
    'LightGBMModel',
    'CatBoostModel',
    'GradientBoostingModel',
    'KNNModel'
]