# models/linear.py
from typing import Dict, Any
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet

from .base import BaseModel


class LinearModel(BaseModel):
    """Linear model with full parameter control"""
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        schema = {
            'fit_intercept': {
                'type': 'bool',
                'default': True,
                'description': 'Whether to fit intercept',
                'category': 'model'
            }
        }
        
        if self.task == "classification":
            schema.update({
                'C': {
                    'type': 'float',
                    'default': 1.0,
                    'min': 0.001,
                    'max': 100.0,
                    'description': 'Inverse regularization strength',
                    'category': 'regularization'
                },
                'solver': {
                    'type': 'str',
                    'default': 'lbfgs',
                    'choices': ['lbfgs', 'liblinear', 'sag', 'saga'],
                    'description': 'Optimization solver',
                    'category': 'optimization'
                },
                'max_iter': {
                    'type': 'int',
                    'default': 1000,
                    'min': 100,
                    'max': 10000,
                    'description': 'Maximum iterations',
                    'category': 'optimization'
                }
            })
        
        return schema
    
    def _create_model(self, **params):
        if self.task == "classification":
            return LogisticRegression(max_iter=1000, **params)
        else:
            return LinearRegression(**params)


class RidgeModel(LinearModel):
    """Ridge regression with L2 regularization"""
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        schema = super().get_parameter_schema()
        schema.update({
            'alpha': {
                'type': 'float',
                'default': 1.0,
                'min': 0.001,
                'max': 100.0,
                'description': 'Regularization strength',
                'category': 'regularization'
            }
        })
        return schema
    
    def _create_model(self, **params):
        return Ridge(**params)


class LassoModel(LinearModel):
    """Lasso regression with L1 regularization"""
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        schema = super().get_parameter_schema()
        schema.update({
            'alpha': {
                'type': 'float',
                'default': 1.0,
                'min': 0.001,
                'max': 100.0,
                'description': 'Regularization strength',
                'category': 'regularization'
            }
        })
        return schema
    
    def _create_model(self, **params):
        return Lasso(max_iter=10000, **params)


class ElasticNetModel(LinearModel):
    """ElasticNet with L1 + L2 regularization"""
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        schema = super().get_parameter_schema()
        schema.update({
            'alpha': {
                'type': 'float',
                'default': 1.0,
                'min': 0.001,
                'max': 100.0,
                'description': 'Regularization strength',
                'category': 'regularization'
            },
            'l1_ratio': {
                'type': 'float',
                'default': 0.5,
                'min': 0.0,
                'max': 1.0,
                'description': 'L1 ratio (0 = L2, 1 = L1)',
                'category': 'regularization'
            }
        })
        return schema
    
    def _create_model(self, **params):
        return ElasticNet(max_iter=10000, **params)