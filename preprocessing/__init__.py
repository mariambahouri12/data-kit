# preprocessing/__init__.py
"""
Preprocessing Package - AI Experimentation Platform

This package provides a collection of preprocessing tools for tabular data.
"""

# ============= Configuration =============
from .tabular.config import PreprocessingConfig # c'est la classe centrale du module , c'est la classe qui contient toutes les configurations

# ============= Version =============
__version__ = '1.0.0'  # version de pachage parent 

# ============= Exports principaux =============
__all__ = [
    'PreprocessingConfig',
]

