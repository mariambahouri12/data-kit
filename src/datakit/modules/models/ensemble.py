# models/ensemble.py - VERSION CORRIGÉE
from .base import BaseModel
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from typing import Dict, Any


class EnsembleModel(BaseModel):
    """Base class for ensemble models with full parameter control"""
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get parameter schema - to be overridden by subclasses"""
        base_schema = {
            'n_estimators': {
                'type': 'int',
                'default': 100,
                'min': 10,
                'max': 1000,
                'description': 'Number of boosting rounds',
                'category': 'boosting'
            },
            'learning_rate': {
                'type': 'float',
                'default': 0.1,
                'min': 0.001,
                'max': 1.0,
                'description': 'Learning rate (step size)',
                'category': 'boosting'
            },
            'random_state': {
                'type': 'int',
                'default': 42,
                'description': 'Random seed',
                'category': 'reproducibility'
            }
        }
        
        return base_schema
    
    def _create_model(self, **params):
        """To be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement _create_model")


# ============= XGBOOST =============

class XGBoostModel(EnsembleModel):
    """XGBoost model"""
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        base_schema = super().get_parameter_schema()
        base_schema.update({
            'max_depth': {
                'type': 'int',
                'default': 6,
                'min': 1,
                'max': 20,
                'description': 'Maximum tree depth',
                'category': 'tree'
            },
            'subsample': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 1.0,
                'description': 'Fraction of samples to use',
                'category': 'sampling'
            },
            'colsample_bytree': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 1.0,
                'description': 'Fraction of features per tree',
                'category': 'sampling'
            },
            'reg_alpha': {
                'type': 'float',
                'default': 0.0,
                'min': 0.0,
                'max': 10.0,
                'description': 'L1 regularization',
                'category': 'regularization'
            },
            'reg_lambda': {
                'type': 'float',
                'default': 1.0,
                'min': 0.0,
                'max': 10.0,
                'description': 'L2 regularization',
                'category': 'regularization'
            }
        })
        return base_schema
    
    def _create_model(self, **params):
        if self.task == "classification":
            return XGBClassifier(**params)
        else:
            return XGBRegressor(**params)


# ============= LIGHTGBM =============

class LightGBMModel(EnsembleModel):
    """LightGBM model"""
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        base_schema = super().get_parameter_schema()
        base_schema.update({
            'num_leaves': {
                'type': 'int',
                'default': 31,
                'min': 2,
                'max': 255,
                'description': 'Maximum number of leaves',
                'category': 'tree'
            },
            'min_child_samples': {
                'type': 'int',
                'default': 20,
                'min': 5,
                'max': 100,
                'description': 'Minimum samples per leaf',
                'category': 'tree'
            },
            'subsample': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 1.0,
                'description': 'Fraction of samples to use',
                'category': 'sampling'
            }
        })
        return base_schema
    
    def _create_model(self, **params):
        if self.task == "classification":
            return LGBMClassifier(**params)
        else:
            return LGBMRegressor(**params)


# ============= CATBOOST =============

class CatBoostModel(EnsembleModel):
    """CatBoost model"""
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        base_schema = super().get_parameter_schema()
        # CatBoost uses 'iterations' instead of 'n_estimators'
        base_schema['n_estimators']['description'] = 'Number of boosting rounds (iterations)'
        
        base_schema.update({
            'depth': {
                'type': 'int',
                'default': 6,
                'min': 1,
                'max': 16,
                'description': 'Tree depth',
                'category': 'tree'
            },
            'l2_leaf_reg': {
                'type': 'float',
                'default': 3.0,
                'min': 0.0,
                'max': 10.0,
                'description': 'L2 regularization',
                'category': 'regularization'
            },
            'border_count': {
                'type': 'int',
                'default': 254,
                'min': 1,
                'max': 255,
                'description': 'Number of discretization bins',
                'category': 'preprocessing'
            }
        })
        return base_schema
    
    def _create_model(self, **params):
        # CatBoost uses 'iterations' but we use 'n_estimators'
        # Convert if needed
        if 'n_estimators' in params:
            params['iterations'] = params.pop('n_estimators')
        
        if self.task == "classification":
            return CatBoostClassifier(verbose=False, **params)
        else:
            return CatBoostRegressor(verbose=False, **params)


# ============= GRADIENT BOOSTING =============

class GradientBoostingModel(EnsembleModel):
    """Gradient Boosting model"""
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        base_schema = super().get_parameter_schema()
        base_schema.update({
            'subsample': {
                'type': 'float',
                'default': 1.0,
                'min': 0.1,
                'max': 1.0,
                'description': 'Fraction of samples to use',
                'category': 'sampling'
            },
            'max_depth': {
                'type': 'int',
                'default': 3,
                'min': 1,
                'max': 20,
                'description': 'Maximum tree depth',
                'category': 'tree'
            },
            'min_samples_split': {
                'type': 'int',
                'default': 2,
                'min': 2,
                'max': 100,
                'description': 'Minimum samples to split',
                'category': 'tree'
            }
        })
        return base_schema
    
    def _create_model(self, **params):
        if self.task == "classification":
            return GradientBoostingClassifier(**params)
        else:
            return GradientBoostingRegressor(**params)