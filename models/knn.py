# models/knn.py
from typing import Dict, Any, Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .base import BaseModel  # ✅ AJOUTÉ


class KNNModel(BaseModel):
    """KNN with sklearn pipeline"""
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Define all configurable parameters"""
        return {
            'n_neighbors': {
                'type': 'int',
                'default': 5,
                'min': 1,
                'max': 100,
                'description': 'Number of neighbors',
                'category': 'model'
            },
            'weights': {
                'type': 'str',
                'default': 'uniform',
                'choices': ['uniform', 'distance'],
                'description': 'Weight function',
                'category': 'model'
            },
            'algorithm': {
                'type': 'str',
                'default': 'auto',
                'choices': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'description': 'Algorithm for computing neighbors',
                'category': 'performance'
            },
            'leaf_size': {
                'type': 'int',
                'default': 30,
                'min': 10,
                'max': 100,
                'description': 'Leaf size for BallTree or KDTree',
                'category': 'performance'
            },
            'metric': {
                'type': 'str',
                'default': 'minkowski',
                'choices': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                'description': 'Distance metric',
                'category': 'distance'
            },
            'p': {
                'type': 'int',
                'default': 2,
                'min': 1,
                'max': 10,
                'description': 'Power parameter for Minkowski metric',
                'category': 'distance'
            },
            'scaler_type': {
                'type': 'str',
                'default': 'standard',
                'choices': ['standard', 'minmax', 'robust', 'none'],
                'description': 'Type of feature scaling',
                'category': 'preprocessing'
            }
        }
    
    def _create_model(self, **params):
        """Create the underlying sklearn KNN model"""
        model_params = {
            k: v for k, v in params.items() 
            if k not in ['scaler_type']
        }
        
        if self.task == "classification":
            return KNeighborsClassifier(**model_params)
        else:
            return KNeighborsRegressor(**model_params)
    
    def _build_pipeline(self) -> Optional[Pipeline]:
        """Build sklearn pipeline with scaling"""
        scaler_type = self.params.get('scaler_type', 'standard')
        
        if scaler_type == 'none':
            return None
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        model_params = {
            k: v for k, v in self.params.items() 
            if k not in ['scaler_type']
        }
        
        model = self._create_model(**model_params)
        
        return Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
    
    def _preprocess(self, X, fit: bool = True):
        """Preprocess data - pipeline handles scaling"""
        return X
    
    def predict_proba(self, X):
        """Probability predictions with pipeline"""
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        if self.pipeline:
            return self.pipeline.predict_proba(X)
        else:
            return self.model.predict_proba(X)
    
    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Find K-neighbors with pipeline"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        if self.pipeline:
            model = self.pipeline.named_steps['model']
            X_transformed = self.pipeline.named_steps['scaler'].transform(X)
        else:
            model = self.model
            X_transformed = X
        
        return model.kneighbors(X_transformed, n_neighbors=n_neighbors, return_distance=return_distance)