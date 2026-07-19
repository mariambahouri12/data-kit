# models/registry.py - VERSION CORRIGÉE
import inspect
import importlib
import pkgutil
from typing import Dict, Any, Type, List, Optional
from .base import BaseModel
import warnings


class ModelRegistry:
    """Registry with static schema support"""
    
    _instance = None
    _models = {}
    _model_schemas = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._discover_models()
        return cls._instance
    
    def _discover_models(self):
        """Auto-discover all model classes"""
        import src.models as models_package
        
        for module_info in pkgutil.iter_modules(models_package.__path__):
            # ✅ Skip internal modules
            if module_info.name.startswith('_'):
                continue
            if module_info.name in ['base', 'registry', 'factory', 'parameter_generator']:
                continue
                
            try:
                module = importlib.import_module(f"models.{module_info.name}")
                
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # ✅ Vérification plus stricte
                    if (issubclass(obj, BaseModel) and 
                        obj != BaseModel and 
                        not inspect.isabstract(obj) and
                        obj.__module__ == module.__name__):  # 👈 Vérifier que la classe est dans le module
                        
                        # Register model
                        model_name = name.replace('Model', '').lower()
                        self._models[model_name] = obj
                        
                        # Get schema safely
                        try:
                            if hasattr(obj, 'get_parameter_schema_static'):
                                schema = obj.get_parameter_schema_static()
                            else:
                                try:
                                    temp_instance = obj(task="classification")
                                    schema = temp_instance.get_parameter_schema()
                                except Exception as e:
                                    warnings.warn(f"Could not get schema for {model_name}: {e}")
                                    schema = {}
                            
                            self._model_schemas[model_name] = schema
                            
                        except Exception as e:
                            warnings.warn(f"Error getting schema for {model_name}: {e}")
                            self._model_schemas[model_name] = {}
                            
            except Exception as e:
                warnings.warn(f"Error loading module {module_info.name}: {e}")
                continue
    
    def get_model(self, model_name: str, task: str = "classification", **params) -> BaseModel:
        """Get a model instance"""
        if model_name not in self._models:
            available = ', '.join(self._models.keys())
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")
        
        model_class = self._models[model_name]
        return model_class(task=task, **params)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with their info"""
        return [
            {
                'name': name,
                'class': cls.__name__,
                'available_tasks': ['classification', 'regression'],
                'parameter_schema': self._model_schemas.get(name, {})
            }
            for name, cls in self._models.items()
        ]
    
    def get_parameter_schema(self, model_name: str) -> Dict[str, Any]:
        """Get parameter schema for a specific model"""
        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self._model_schemas.get(model_name, {})
    
    def get_model_class(self, model_name: str) -> Optional[Type[BaseModel]]:
        """Get model class without instantiating"""
        return self._models.get(model_name)
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self._models.keys())
    
    def clear_cache(self):
        """Clear the registry cache (useful for testing)"""
        self._models = {}
        self._model_schemas = {}
        self._discover_models()