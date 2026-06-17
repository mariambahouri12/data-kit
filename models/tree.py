# models/tree.py
from typing import Dict, Any
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .base import BaseModel  # ✅ AJOUTÉ


class TreeModel(BaseModel):
    """Tree-based model with full parameter control"""
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        base_schema = {
            'max_depth': {
                'type': 'int',
                'default': None,
                'min': 1,
                'max': 100,
                'description': 'Maximum tree depth (None = unlimited)',
                'category': 'tree'
            },
            'min_samples_split': {
                'type': 'int',
                'default': 2,
                'min': 2,
                'max': 100,
                'description': 'Minimum samples to split a node',
                'category': 'tree'
            },
            'min_samples_leaf': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 100,
                'description': 'Minimum samples per leaf',
                'category': 'tree'
            },
            'max_features': {
                'type': 'str',
                'default': 'sqrt',
                'choices': ['auto', 'sqrt', 'log2', None],
                'description': 'Number of features to consider',
                'category': 'tree'
            },
            'random_state': {
                'type': 'int',
                'default': 42,
                'description': 'Random seed',
                'category': 'reproducibility'
            }
        }
        
        # Additional params for Random Forest
        if 'randomforest' in self.__class__.__name__.lower():
            base_schema.update({
                'n_estimators': {
                    'type': 'int',
                    'default': 100,
                    'min': 10,
                    'max': 1000,
                    'description': 'Number of trees',
                    'category': 'ensemble'
                },
                'bootstrap': {
                    'type': 'bool',
                    'default': True,
                    'description': 'Use bootstrap samples',
                    'category': 'ensemble'
                },
                'n_jobs': {
                    'type': 'int',
                    'default': -1,
                    'min': -1,
                    'max': 16,
                    'description': 'Number of parallel jobs (-1 = all cores)',
                    'category': 'performance'
                }
            })
        
        return base_schema
    
    def _create_model(self, **params):
        model_name = self.__class__.__name__.lower()
        
        if self.task == "classification":
            if 'randomforest' in model_name:
                return RandomForestClassifier(**params)
            else:
                return DecisionTreeClassifier(**params)
        else:
            if 'randomforest' in model_name:
                return RandomForestRegressor(**params)
            else:
                return DecisionTreeRegressor(**params)


class RandomForestModel(TreeModel):
    """Random Forest model (wrapper for TreeModel with RF params)"""
    
    def __init__(self, task: str = "classification", **kwargs):
        super().__init__(task=task, **kwargs)