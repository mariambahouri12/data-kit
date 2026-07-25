# tracking/__init__.py
"""
AI Experimentation Platform - Tracking Package

Suivi des expériences (runs, paramètres, métriques, versions de modèles).
Séparé du package `models` : c'est une préoccupation de persistance/MLOps,
pas de modélisation.
"""

from .experiment_tracker import ExperimentTracker

__all__ = ["ExperimentTracker"]
