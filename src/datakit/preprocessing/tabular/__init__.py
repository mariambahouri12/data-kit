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


# ============= Configuration =============
from datakit.preprocessing.tabular.transformers import PercentileTransformer
from .transformers import boxcox, log, reciprocal, sqrt

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
from .encoders.encoders import (
    CategoricalEncoder,
  
)

# ============= Scaling =============
from .transformers.scalers import (
    FeatureScaler,
  
)

# ============= Transformations =============
from .transformers import (
    yeojohnson
)

# ============= Réduction =============
from .reducers import (
    FeatureSelector,
    PCAReducer,
    LDAReducer
)

# ============= Rééquilibrage =============
from .balancers.balancers import ClassBalancer

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
__version__ = '1.0.0'  # Format : MAJEUR.MINEUR.PATCH


# ============= Exports =============
__all__ = [  # Liste des noms exportés lors d'un `from preprocessing.tabular import *`
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
 

    # Scaling
    'FeatureScaler',


    # Transformations
    'log',
    'sqrt',
    'reciprocal',
    'boxcox',
    'yeojohnson',
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
]


# ============= Documentation =============
def info() -> None:
    """Afficher les informations sur le module."""
    print("=" * 60)
    print("📊 TABULAR PREPROCESSING MODULE")
    print("=" * 60)
    print(f"Version: {__version__}")
    print()
    print("📦 Sous-modules disponibles:")
    print("  • config              - Configuration du prétraitement")
    print("  • detectors           - Détection des problèmes")
    print("  • cleaners            - Nettoyage des données")
    print("  • encoders            - Encodage des catégories")
    print("  • scalers             - Normalisation/Standardisation")
    print("  • transformers        - Transformations de distribution")
    print("  • reducers            - Réduction de dimensionnalité")
    print("  • balancers           - Rééquilibrage des classes")
    print("  • feature_engineering - Création de features")
    print("  • pipeline_builder    - Construction de pipelines")
    print()
    print("📖 Documentation : voir le README du dépôt GitHub")
    print("=" * 60)