# preprocessing/__init__.py
"""
Preprocessing Package - AI Experimentation Platform

Ce package fournit les outils de prétraitement pour données tabulaires.
"""

# ============= Configuration =============
from .tabular.config import PreprocessingConfig

# ============= Version =============
__version__ = '1.0.0'

# ============= Exports principaux =============
__all__ = [
    'PreprocessingConfig',
]

# ============= Documentation =============
def info():
    """Afficher les informations sur le module."""
    print("=" * 60)
    print("📊 TABULAR PREPROCESSING MODULE")
    print("=" * 60)
    print(f"Version: {__version__}")
    print()
    print("📦 Importez les composants individuellement :")
    print("  from preprocessing.tabular.encoders import CategoricalEncoder")
    print("  from preprocessing.tabular.scalers import FeatureScaler")
    print("  from preprocessing.tabular.cleaners import MissingValueCleaner")
    print("=" * 60)