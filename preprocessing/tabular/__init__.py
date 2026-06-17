# preprocessing/tabular/__init__.py
"""
Tabular Preprocessing Module - Prétraitement des Données Tabulaire

Ce module fournit tous les outils nécessaires pour le prétraitement
des données tabulaires (CSV, Excel, etc.) :

- Détection des problèmes (valeurs manquantes, outliers, corrélations)
- Nettoyage des données
- Encodage des variables catégorielles
- Normalisation/Standardisation
- Transformations de distribution
- Réduction de dimensionnalité
- Rééquilibrage des classes
- Feature Engineering
- Construction de pipelines
"""

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

# ============= Upload =============
from .upload import CSVUploader


# ============= Version =============
__version__ = '1.0.0'


# ============= Exports =============
__all__ = [
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


# ============= Documentation =============
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
    print("📖 Documentation: https://github.com/your-repo")
    print("=" * 60)


# ============= Test rapide =============
def test():
    """
    Test rapide du module.
    """
    import pandas as pd
    import numpy as np
    
    print("🧪 Testing Tabular Preprocessing Module...")
    
    # Créer des données de test
    np.random.seed(42)
    df = pd.DataFrame({
        'num1': np.random.randn(100),
        'num2': np.random.randn(100) * 2 + 5,
        'cat1': np.random.choice(['A', 'B', 'C'], 100),
        'cat2': np.random.choice(['X', 'Y', 'Z', 'W'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    
    # Ajouter des valeurs manquantes
    df.loc[0:10, 'num1'] = np.nan
    df.loc[20:25, 'cat1'] = np.nan
    
    print(f"✅ Données de test créées: {df.shape}")
    
    # Tester les détecteurs
    from .detectors import MissingValueDetector, OutlierDetector
    
    missing_detector = MissingValueDetector()
    missing_detector.fit(df)
    print(f"✅ MissingValueDetector: {len(missing_detector.problems)} problèmes")
    
    outlier_detector = OutlierDetector()
    outlier_detector.fit(df)
    print(f"✅ OutlierDetector: {len(outlier_detector.problems)} problèmes")
    
    # Tester l'encodage
    from .encoders import CategoricalEncoder
    
    encoder = CategoricalEncoder(method='onehot', columns=['cat1', 'cat2'])
    encoder.fit(df)
    df_encoded = encoder.transform(df)
    print(f"✅ CategoricalEncoder: {df_encoded.shape}")
    
    # Tester le scaling
    from .scalers import FeatureScaler
    
    scaler = FeatureScaler(method='standard', columns=['num1', 'num2'])
    scaler.fit(df)
    df_scaled = scaler.transform(df)
    print(f"✅ FeatureScaler: {df_scaled.shape}")
    
    print("\n🎉 Tous les tests sont passés !")
    print("📊 Module prêt à l'emploi.")
    
    return True


# Exécuter le test si le fichier est exécuté directement
if __name__ == "__main__":
    test()