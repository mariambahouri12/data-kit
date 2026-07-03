# preprocessing/tabular/__init__.py
"""
Tabular Preprocessing Module - Tabular Data Preprocessing

This module provides all the necessary tools for preprocessing
tabular data (CSV, Excel, etc.):

- Issue detection (missing values, outliers, correlations)
- Data cleaning
- Categorical variable encoding
- Normalization/Standardization
- Distribution transformations
- Dimensionality reduction
- Class balancing
- Feature engineering
- Construction of pipelines
"""
# ce fichier est le point d'entrée principal du module de prétraitmenr tabulaire , il sert à organiser et structuere le module en regroupant tous les sous-modules , facilier les imports en permettant d'importer directement depuis ppreprocessing.tabular, exposer l'API publique du module (ce qui accessible aux utilisateurs) , fournir de la documentation 
# ============= Configuration =============
from .config import (
    PreprocessingConfig,
    ImputationMethod,
    ScalingMethod,
    EncodingMethod,
    OutlierMethod,
    OutlierAction,
    BalancingMethod,
    FeatureSelectionMethod
)

# ============= Détecteurs =============
from .detectors import (
    MissingValueDetector,
    OutlierDetector,
    CorrelationDetector,
    CardinalityDetector,
    DuplicateDetector
)

# ============= Nettoyage =============
from .cleaners import (
    MissingValueCleaner,
    OutlierCleaner,
    DuplicateCleaner
)

# ============= Encodage =============
from .encoders import (
    CategoricalEncoder,
    OrdinalEncoderWrapper
)

# ============= Scaling =============
from .scalers import (
    FeatureScaler,
    PowerTransformerWrapper
)

# ============= Transformations =============
from .transformers import (
    LogTransformer,
    SqrtTransformer,
    ReciprocalTransformer,
    BoxCoxTransformer,
    YeoJohnsonTransformer,
    PercentileTransformer
)

# ============= Réduction =============
from .reducers import (
    FeatureSelector,
    PCAReducer,
    LDAReducer
)

# ============= Rééquilibrage =============
from .balancers import ClassBalancer

# ============= Feature Engineering =============
from .feature_engineering import (
    PolynomialFeatureCreator,
    InteractionFeatureCreator,
    RatioFeatureCreator,
    AggregationFeatureCreator,
    DateFeatureCreator
)

# ============= Pipeline =============
from .pipeline_builder import (
    PipelineBuilder,
    SimplePipelineBuilder
)


# ============= Version =============
__version__ = '1.0.0' # c'est la version de mon module  MAJUER.MINEUR.PATH 


# ============= Exports =============
__all__ = [  # liste des noms qui seront exporté quant on fait from preprocessing.tabular import *
    # Configuration
    'PreprocessingConfig',
    'ImputationMethod',
    'ScalingMethod',
    'EncodingMethod',
    'OutlierMethod',
    'OutlierAction',
    'BalancingMethod',
    'FeatureSelectionMethod',
    
    # Détecteurs
    'MissingValueDetector',
    'OutlierDetector',
    'CorrelationDetector',
    'CardinalityDetector',
    'DuplicateDetector',
    
    # Nettoyage
    'MissingValueCleaner',
    'OutlierCleaner',
    'DuplicateCleaner',
    
    # Encodage
    'CategoricalEncoder',
    'OrdinalEncoderWrapper',
    
    # Scaling
    'FeatureScaler',
    'PowerTransformerWrapper',
    
    # Transformations
    'LogTransformer',
    'SqrtTransformer',
    'ReciprocalTransformer',
    'BoxCoxTransformer',
    'YeoJohnsonTransformer',
    'PercentileTransformer',
    
    # Réduction
    'FeatureSelector',
    'PCAReducer',
    'LDAReducer',
    
    # Rééquilibrage
    'ClassBalancer',
    
    # Feature Engineering
    'PolynomialFeatureCreator',
    'InteractionFeatureCreator',
    'RatioFeatureCreator',
    'AggregationFeatureCreator',
    'DateFeatureCreator',
    
    # Pipeline
    'PipelineBuilder',
    'SimplePipelineBuilder',
    
    # Upload
    'CSVUploader'
]


# ============= Documentation ============= # cest pour afficher les informations sur le module
def info():
    """
    Afficher les informations sur le module.
    """
    print("=" * 60)
    print("📊 TABULAR PREPROCESSING MODULE")
    print("=" * 60)
    print(f"Version: {__version__}")
    print()
    print("📦 Sous-modules disponibles:")
    print("  • config          - Configuration du prétraitement")
    print("  • detectors       - Détection des problèmes")
    print("  • cleaners        - Nettoyage des données")
    print("  • encoders        - Encodage des catégories")
    print("  • scalers         - Normalisation/Standardisation")
    print("  • transformers    - Transformations de distribution")
    print("  • reducers        - Réduction de dimensionnalité")
    print("  • balancers       - Rééquilibrage des classes")
    print("  • feature_engineering - Création de features")
    print("  • pipeline_builder - Construction de pipelines")
    print("  • upload          - Upload de fichiers")
    print()
    print("📖 Documentation: tw nzidou baad github repo")
    print("=" * 60)

