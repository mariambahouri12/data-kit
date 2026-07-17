# preprocessing/factory.py
from typing import Dict, Any, Optional, List, Type, Union
import pandas as pd
from sklearn.pipeline import Pipeline

from .base import BasePreprocessor
from .tabular.config import PreprocessingConfig, TaskType
from .tabular.pipeline_builder import PipelineBuilder, SimplePipelineBuilder
from .tabular.detectors import (
    MissingValueDetector, OutlierDetector, CorrelationDetector,
    CardinalityDetector, DuplicateDetector
)
from .tabular.cleaners import MissingValueCleaner, OutlierCleaner, DuplicateCleaner
from .tabular.encoders import CategoricalEncoder
from .tabular.scalers import FeatureScaler
from .tabular.transformers import (
    LogTransformer, SqrtTransformer, BoxCoxTransformer,
    YeoJohnsonTransformer, PercentileTransformer
)
from .tabular.reducers import FeatureSelector, PCAReducer, LDAReducer
from .tabular.balancers import ClassBalancer
from .tabular.feature_engineering import (
    PolynomialFeatureCreator, InteractionFeatureCreator,
    RatioFeatureCreator, AggregationFeatureCreator, DateFeatureCreator
)


class PreprocessingFactory:
    """
    Factory pour créer des préprocesseurs et pipelines.
    Permet de créer des composants individuels ou des pipelines complets.
    """
    
    # Registre des préprocesseurs disponibles
    _preprocessors = {
        # Détecteurs
        'missing_detector': MissingValueDetector,
        'outlier_detector': OutlierDetector,
        'correlation_detector': CorrelationDetector,
        'cardinality_detector': CardinalityDetector,
        'duplicate_detector': DuplicateDetector,
        
        # Nettoyage
        'missing_cleaner': MissingValueCleaner,
        'outlier_cleaner': OutlierCleaner,
        'duplicate_cleaner': DuplicateCleaner,
        
        # Encodage
        'categorical_encoder': CategoricalEncoder,

        
        # Scaling
        'feature_scaler': FeatureScaler,

        
        # Transformations
        'log_transformer': LogTransformer,
        'sqrt_transformer': SqrtTransformer,
        'boxcox_transformer': BoxCoxTransformer,
        'yeojohnson_transformer': YeoJohnsonTransformer,
        'percentile_transformer': PercentileTransformer,
        
        # Réduction
        'feature_selector': FeatureSelector,
        'pca_reducer': PCAReducer,
        'lda_reducer': LDAReducer,
        
        # Rééquilibrage
        'class_balancer': ClassBalancer,
        
        # Feature Engineering
        'polynomial_creator': PolynomialFeatureCreator,
        'interaction_creator': InteractionFeatureCreator,
        'ratio_creator': RatioFeatureCreator,
        'aggregation_creator': AggregationFeatureCreator,
        'date_creator': DateFeatureCreator
    }
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BasePreprocessor:
        """
        Créer un préprocesseur par son nom.
        
        Args:
            name: Nom du préprocesseur
            **kwargs: Paramètres du préprocesseur
        
        Returns:
            Instance du préprocesseur
        """
        if name not in cls._preprocessors:
            available = ', '.join(cls._preprocessors.keys())
            raise ValueError(f"Preprocessor '{name}' not found. Available: {available}")
        
        preprocessor_class = cls._preprocessors[name]
        return preprocessor_class(**kwargs)
    
    @classmethod
    def list_preprocessors(cls) -> List[Dict[str, Any]]:
        """
        Lister tous les préprocesseurs disponibles.
        
        Returns:
            Liste des préprocesseurs avec leurs informations
        """
        return [
            {
                'name': name,
                'class': cls.__name__,
                'description': cls.__doc__.strip() if cls.__doc__ else ''
            }
            for name, cls in cls._preprocessors.items()
        ]
    
    @classmethod
    def create_pipeline(cls, 
                        config: Optional[Union[Dict[str, Any], PreprocessingConfig]] = None,
                        X: Optional[pd.DataFrame] = None,
                        y: Optional[pd.Series] = None) -> Pipeline:
        """
        Créer un pipeline complet.
        
        Args:
            config: Configuration (dictionnaire ou PreprocessingConfig)
            X: DataFrame pour détection automatique
            y: Target pour Target Encoding
        
        Returns:
            Pipeline sklearn
        """
        if isinstance(config, dict):
            config = PreprocessingConfig.from_dict(config)
        elif config is None:
            config = PreprocessingConfig()
        
        builder = PipelineBuilder(config)
        return builder.build_pipeline(X, y)
    
    @classmethod
    def create_detection_pipeline(cls) -> Pipeline:
        """Créer un pipeline de détection"""
        builder = PipelineBuilder()
        return builder.build_detection_pipeline()
    
    @classmethod
    def create_simple_pipeline(cls, 
                               pipeline_type: str = 'default',
                               task_type: str = 'classification',
                               **kwargs) -> Pipeline:
        """
        Créer un pipeline simple prédéfini.
        
        Args:
            pipeline_type: 'default', 'robust', 'high_performance', 'minimal'
            task_type: 'classification' ou 'regression'
            **kwargs: Paramètres supplémentaires
        
        Returns:
            Pipeline sklearn
        """
        kwargs['task_type'] = TaskType(task_type)
        
        if pipeline_type == 'default':
            builder = SimplePipelineBuilder.create_default(**kwargs)
        elif pipeline_type == 'robust':
            builder = SimplePipelineBuilder.create_robust(**kwargs)
        elif pipeline_type == 'high_performance':
            builder = SimplePipelineBuilder.create_high_performance(**kwargs)
        elif pipeline_type == 'minimal':
            builder = SimplePipelineBuilder.create_minimal(**kwargs)
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        return builder.build_pipeline()
    
    @classmethod
    def get_preprocessor_info(cls, name: str) -> Dict[str, Any]:
        """
        Obtenir les informations d'un préprocesseur.
        
        Args:
            name: Nom du préprocesseur
        
        Returns:
            Informations du préprocesseur
        """
        if name not in cls._preprocessors:
            raise ValueError(f"Preprocessor '{name}' not found")
        
        preprocessor_class = cls._preprocessors[name]
        
        return {
            'name': name,
            'class': preprocessor_class.__name__,
            'description': preprocessor_class.__doc__.strip() if preprocessor_class.__doc__ else '',
        }
    
    @classmethod
    def register_preprocessor(cls, name: str, preprocessor_class: Type[BasePreprocessor]):
        """
        Enregistrer un nouveau préprocesseur.
        
        Args:
            name: Nom du préprocesseur
            preprocessor_class: Classe du préprocesseur
        """
        if not issubclass(preprocessor_class, BasePreprocessor):
            raise ValueError(f"{preprocessor_class.__name__} must inherit from BasePreprocessor")
        
        cls._preprocessors[name] = preprocessor_class


# ============= Préréglages de Configurations =============

class PreprocessingPresets:
    """Préréglages de configurations pour différents cas d'usage"""
    
    @staticmethod
    def get_preset(name: str) -> Dict[str, Any]:
        """
        Obtenir un préréglage de configuration.
        
        Args:
            name: Nom du préréglage
        
        Returns:
            Dictionnaire de configuration
        """
        presets = {
            'quick': {
                'imputation_method': 'median',
                'scaling_method': 'standard',
                'encoding_method': 'onehot',
                'outlier_method': 'iqr',
                'outlier_threshold': 1.5
            },
            'robust': {
                'imputation_method': 'median',
                'scaling_method': 'robust',
                'encoding_method': 'target',
                'outlier_method': 'iqr',
                'outlier_threshold': 3.0,
                'outlier_action': 'winsorize'
            },
            'high_performance': {
                'imputation_method': 'knn',
                'scaling_method': 'standard',
                'encoding_method': 'catboost',
                'outlier_method': 'isolation_forest',
                'outlier_action': 'winsorize',
                'create_polynomial': True,
                'polynomial_degree': 2,
                'apply_boxcox': True
            },
            'minimal': {
                'imputation_method': 'median',
                'scaling_method': 'standard',
                'encoding_method': 'ordinal',
                'outlier_method': 'none',
                'drop_duplicates': False,
                'drop_high_missing': False
            },
            'nlp_ready': {
                'imputation_method': 'most_frequent',
                'scaling_method': 'standard',
                'encoding_method': 'frequency',
                'outlier_method': 'iqr',
                'outlier_threshold': 1.5,
                'create_polynomial': False
            },
            'time_series': {
                'imputation_method': 'median',
                'scaling_method': 'standard',
                'encoding_method': 'onehot',
                'outlier_method': 'iqr',
                'outlier_threshold': 1.5,
                'drop_duplicates': True,
                'create_interactions': False,
                'create_ratios': False
            },
            'imbalanced': {
                'imputation_method': 'median',
                'scaling_method': 'standard',
                'encoding_method': 'target',
                'outlier_method': 'iqr',
                'outlier_threshold': 1.5,
                'balancing_method': 'smote',
                'balancing_apply_before_pipeline': True,
                'outlier_action': 'winsorize'
            }
        }
        
        if name not in presets:
            available = ', '.join(presets.keys())
            raise ValueError(f"Preset '{name}' not found. Available: {available}")
        
        return presets[name]
    
    @staticmethod
    def list_presets() -> List[str]:
        """Lister tous les préréglages disponibles"""
        return [
            'quick', 'robust', 'high_performance', 'minimal',
            'nlp_ready', 'time_series', 'imbalanced'
        ]