# models/base.py - VERSION FINALE CORRIGÉE
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
from sklearn.pipeline import Pipeline
from datetime import datetime
import uuid
import json
import warnings
import joblib  # ✅ AJOUTÉ
import os      # ✅ AJOUTÉ


class BaseModel(ABC):
    """Base class with separation of concerns"""
    
    def __init__(self, task: str = "classification", **kwargs):
        self.task = task
        self.params = self.get_default_params()
        self.params.update(kwargs)
        self.model = None
        self.pipeline = None
        self.is_trained = False
        self.metrics = {}
        self.run_id = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        self.training_history = []
        self.best_params = {}
        self.best_score = -float('inf')
        self.cv_results = None  # ✅ AJOUTÉ
        self.experiment_data = {}  # ✅ AJOUTÉ
        
        # Validate parameters
        self._validate_params(self.params)
    
    @abstractmethod
    def _create_model(self, **params):
        """Create the underlying model"""
        pass
    
    @abstractmethod
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Return parameter schema"""
        pass
    
    @staticmethod
    def get_parameter_schema_static() -> Dict[str, Any]:
        """Static method for registry discovery without instantiation"""
        return {}
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters from schema"""
        schema = self.get_parameter_schema()
        return {k: v['default'] for k, v in schema.items()}
    
    def _validate_params(self, params: Dict[str, Any], strict: bool = False) -> bool:
        """
        Validate parameters against schema.
        
        Args:
            params: Parameters to validate
            strict: If True, raise error for unknown params.
                    If False, only warn.
        """
        schema = self.get_parameter_schema()
        
        for param_name, param_value in params.items():
            if param_name not in schema:
                if strict:
                    raise ValueError(f"Unknown parameter: {param_name}")
                else:
                    warnings.warn(f"Unknown parameter '{param_name}' ignored")
                continue
            
            param_def = schema[param_name]
            
            # Skip validation if value is None
            if param_value is None:  # ✅ AJOUTÉ
                continue
            
            # Type validation
            if param_def['type'] == 'int':
                if not isinstance(param_value, int):
                    raise TypeError(f"{param_name} must be int, got {type(param_value)}")
                if 'min' in param_def and param_value < param_def['min']:
                    raise ValueError(f"{param_name} must be >= {param_def['min']}")
                if 'max' in param_def and param_value > param_def['max']:
                    raise ValueError(f"{param_name} must be <= {param_def['max']}")
                    
            elif param_def['type'] == 'float':
                if not isinstance(param_value, (int, float)):
                    raise TypeError(f"{param_name} must be float, got {type(param_value)}")
                if 'min' in param_def and param_value < param_def['min']:
                    raise ValueError(f"{param_name} must be >= {param_def['min']}")
                if 'max' in param_def and param_value > param_def['max']:
                    raise ValueError(f"{param_name} must be <= {param_def['max']}")
                    
            elif param_def['type'] == 'str':
                if not isinstance(param_value, str):
                    raise TypeError(f"{param_name} must be str, got {type(param_value)}")
                if 'choices' in param_def and param_value not in param_def['choices']:
                    raise ValueError(f"{param_name} must be one of {param_def['choices']}")
                    
            elif param_def['type'] == 'bool':
                if not isinstance(param_value, bool):
                    raise TypeError(f"{param_name} must be bool, got {type(param_value)}")
        
        return True
    
    # ============= SEPARATION OF CONCERNS =============
    
    def fit(self, X, y, **fit_params):
        """
        Pure fit method - just trains the model on data.
        No splitting, no evaluation.
        """
        # Preprocess
        X_processed = self._preprocess(X, fit=True)
        
        # Create model with current params
        self.model = self._create_model(**self.params)
        
        # Build pipeline if preprocessing is needed
        self.pipeline = self._build_pipeline()
        
        # Fit
        if self.pipeline:
            self.pipeline.fit(X_processed, y, **fit_params)
        else:
            self.model.fit(X_processed, y, **fit_params)
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        """Pure predict - no evaluation"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_processed = self._preprocess(X, fit=False)
        
        if self.pipeline:
            return self.pipeline.predict(X_processed)
        else:
            return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        """Probability predictions (classification only)"""
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        X_processed = self._preprocess(X, fit=False)
        
        if self.pipeline:
            return self.pipeline.predict_proba(X_processed)
        else:
            return self.model.predict_proba(X_processed)
    
    def evaluate(self, X, y, metrics: Optional[List[str]] = None):
        """
        Pure evaluate - just computes metrics on given data.
        No fitting, no splitting.
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        y_pred = self.predict(X)
        return self._compute_metrics(y, y_pred, metrics)
    
    def train(self, X, y, test_size: float = 0.2, random_state: int = 42, **fit_params):
        """
        Full train-evaluate workflow.
        Uses fit() + evaluate() internally.
        """
        # Split data
        X_train, X_test, y_train, y_test = self._split_data(
            X, y, test_size, random_state
        )
        
        # Fit
        self.fit(X_train, y_train, **fit_params)
        
        # Evaluate
        self.metrics = self.evaluate(X_test, y_test)
        self.metrics['test_size'] = len(y_test)
        self.metrics['train_size'] = len(X_train)
        
        # Log
        self._log_experiment()
        
        return self.metrics
    
    def cross_validate(self, X, y, cv: int = 5, scoring: Optional[str] = None):
        """
        Cross-validation evaluation.
        """
        if scoring is None:
            scoring = 'accuracy' if self.task == 'classification' else 'r2'
        
        X_processed = self._preprocess(X, fit=True)
        
        # Create pipeline if needed
        pipeline = self._build_pipeline()
        
        if pipeline:
            cv_results = cross_validate(
                pipeline, X_processed, y, 
                cv=cv, scoring=scoring, 
                return_train_score=True
            )
        else:
            # Build model
            model = self._create_model(**self.params)
            cv_results = cross_validate(
                model, X_processed, y, 
                cv=cv, scoring=scoring,
                return_train_score=True
            )
        
        # Store results
        self.cv_results = {
            'cv_scores': cv_results['test_score'],
            'mean_score': cv_results['test_score'].mean(),
            'std_score': cv_results['test_score'].std(),
            'train_scores': cv_results['train_score'].mean()
        }
        
        return self.cv_results
    
    # ============= PREPROCESSING =============
    
    def _build_pipeline(self) -> Optional[Pipeline]:
        """
        Build sklearn pipeline if preprocessing is needed.
        Override in child classes.
        """
        return None
    
    def _preprocess(self, X, fit: bool = True):
        """
        Preprocess data before fit/predict.
        Override in child classes.
        """
        return X
    
    def _split_data(self, X, y, test_size: float = 0.2, random_state: int = 42):
        """Split data for training"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # ============= METRICS =============
    
    def _compute_metrics(self, y_true, y_pred, metrics: Optional[List[str]] = None):
        """Compute metrics based on task"""
        all_metrics = {}
        
        if self.task == "classification":
            # Classification metrics
            all_metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            try:
                all_metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
                all_metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
            except:
                pass
                
        else:  # regression
            all_metrics['mse'] = mean_squared_error(y_true, y_pred)
            all_metrics['rmse'] = np.sqrt(all_metrics['mse'])
            all_metrics['r2'] = r2_score(y_true, y_pred)
        
        # Filter if specific metrics requested
        if metrics:
            return {k: v for k, v in all_metrics.items() if k in metrics}
        
        return all_metrics
    
    # ============= EXPERIMENT TRACKING =============
    
    def _log_experiment(self):
        """Log experiment data"""
        self.experiment_data = {
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'model_name': self.__class__.__name__,
            'task': self.task,
            'params': self.params,
            'metrics': self.metrics,
            'cv_results': getattr(self, 'cv_results', None)
        }
        
        # Add to training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'params': self.params.copy(),
            'metrics': self.metrics.copy()
        })
    
    def get_experiment_data(self) -> Dict[str, Any]:
        """Get experiment data for tracking"""
        return getattr(self, 'experiment_data', {})
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history"""
        return self.training_history
    
    # ============= UTILITY =============
    
    def set_params(self, **params):
        """Set new parameters"""
        self.params.update(params)
        self._validate_params(self.params)
        self.model = None
        self.pipeline = None
        self.is_trained = False
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Get current parameters"""
        return self.params.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics"""
        return self.metrics.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializable representation"""
        return {
            'run_id': self.run_id,
            'model_name': self.__class__.__name__,
            'task': self.task,
            'params': self.params,
            'metrics': self.metrics,
            'is_trained': self.is_trained,
            'timestamp': self.timestamp
        }
    
    def save(self, path: str):
        """Save model and metadata"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model!")
        
        # Create directory
        os.makedirs(path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(path, f"{self.run_id}_model.pkl")
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata_path = os.path.join(path, f"{self.run_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Save pipeline if exists
        if self.pipeline:
            pipeline_path = os.path.join(path, f"{self.run_id}_pipeline.pkl")
            joblib.dump(self.pipeline, pipeline_path)
        
        return {
            'model_path': model_path,
            'metadata_path': metadata_path,
            'run_id': self.run_id
        }
    
    @classmethod
    def load(cls, path: str, run_id: str):
        """Load a saved model"""
        model_path = os.path.join(path, f"{run_id}_model.pkl")
        metadata_path = os.path.join(path, f"{run_id}_metadata.json")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(task=metadata['task'], **metadata['params'])
        
        # Load model
        instance.model = joblib.load(model_path)
        instance.is_trained = True
        instance.metrics = metadata['metrics']
        instance.run_id = run_id
        
        # Load pipeline if exists
        pipeline_path = os.path.join(path, f"{run_id}_pipeline.pkl")
        if os.path.exists(pipeline_path):
            instance.pipeline = joblib.load(pipeline_path)
        
        return instance