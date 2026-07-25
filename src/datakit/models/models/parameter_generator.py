# models/parameter_generator.py
import random
from typing import Dict, Any, List
import numpy as np

class ParameterGenerator:
    """Generate parameters for AutoML"""
    
    @staticmethod
    def generate_random_params(schema: Dict[str, Any], n_samples: int = 1) -> List[Dict[str, Any]]:
        """Generate random parameters for hyperparameter search"""
        param_sets = []
        
        for _ in range(n_samples):
            params = {}
            for param_name, param_def in schema.items():
                param_type = param_def['type']
                
                if param_type == 'int':
                    if 'choices' in param_def:
                        params[param_name] = random.choice(param_def['choices'])
                    elif 'min' in param_def and 'max' in param_def:
                        params[param_name] = random.randint(param_def['min'], param_def['max'])
                    else:
                        params[param_name] = param_def['default']
                
                elif param_type == 'float':
                    if 'choices' in param_def:
                        params[param_name] = random.choice(param_def['choices'])
                    elif 'min' in param_def and 'max' in param_def:
                        params[param_name] = random.uniform(param_def['min'], param_def['max'])
                    else:
                        params[param_name] = param_def['default']
                
                elif param_type == 'str':
                    if 'choices' in param_def:
                        params[param_name] = random.choice(param_def['choices'])
                    else:
                        params[param_name] = param_def['default']
                
                elif param_type == 'bool':
                    params[param_name] = random.choice([True, False])
            
            param_sets.append(params)
        
        return param_sets
    
    @staticmethod
    def generate_grid_params(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate grid parameters for exhaustive search"""
        # Simple implementation - expand for production
        return [ParameterGenerator.generate_random_params(schema, 1)[0]]
    
    @staticmethod
    def suggest_optimal_params(schema: Dict[str, Any], data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest optimal parameters based on data info"""
        params = {}
        
        for param_name, param_def in schema.items():
            # Use default by default
            params[param_name] = param_def['default']
            
            # Some heuristics
            if param_name == 'n_estimators':
                # More data = more trees
                n_samples = data_info.get('n_samples', 1000)
                if n_samples > 10000:
                    params[param_name] = min(500, param_def.get('max', 500))
                elif n_samples > 5000:
                    params[param_name] = 200
                    
            elif param_name == 'max_depth':
                # More features = deeper trees
                n_features = data_info.get('n_features', 10)
                if n_features > 50:
                    params[param_name] = min(10, param_def.get('max', 20))
                    
            elif param_name == 'learning_rate':
                # More data = lower learning rate
                n_samples = data_info.get('n_samples', 1000)
                if n_samples > 10000:
                    params[param_name] = 0.05
                    
            elif param_name == 'alpha' and 'regularization' in param_def.get('category', ''):
                # More features = more regularization
                n_features = data_info.get('n_features', 10)
                if n_features > 100:
                    params[param_name] = min(2.0, param_def.get('max', 10.0))
        
        return params
