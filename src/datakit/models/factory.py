# models/factory.py
from typing import Dict, Any, Optional
from .registry import ModelRegistry

class ModelFactory:
    """Factory for creating models with flexible configuration"""
    
    def __init__(self):
        self.registry = ModelRegistry()
    
    def create_model(
        self,
        model_name: str,
        task: str = "classification",
        user_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Create a model with user-specified parameters
        
        Args:
            model_name: Name of the model
            task: 'classification' or 'regression'
            user_params: User-specified parameters
            **kwargs: Additional parameters
        
        Returns:
            Configured model instance
        """
        # Merge parameters (user_params override kwargs)
        params = kwargs.copy()
        if user_params:
            params.update(user_params)
        
        # Validate parameters
        schema = self.registry.get_parameter_schema(model_name)
        
        # Remove invalid parameters
        valid_params = {k: v for k, v in params.items() if k in schema}
        
        # Add default parameters if missing
        for param_name, param_def in schema.items():
            if param_name not in valid_params:
                valid_params[param_name] = param_def['default']
        
        # Create model
        return self.registry.get_model(model_name, task=task, **valid_params)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model"""
        schema = self.registry.get_parameter_schema(model_name)
        
        return {
            'name': model_name,
            'parameters': schema,
            'available_tasks': ['classification', 'regression']
        }