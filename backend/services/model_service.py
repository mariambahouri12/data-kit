"""
Model service - utilise datakit.models.
"""

import pandas as pd
from typing import Dict, Any, List, Optional

from .state import data_state


class ModelService:
    """Service pour l'entraînement des modèles."""

    def __init__(self):
        self._model = None
        self._metrics = None

    def get_available_models(self) -> List[str]:
        """Liste des modèles disponibles."""
        try:
            from datakit.models.models.registry import ModelRegistry
            registry = ModelRegistry()
            return registry.get_available_models()
        except ImportError:
            return ["RandomForest", "XGBoost", "LogisticRegression", "KNN", "SVM"]

    def get_parameter_schema(self, model_name: str) -> Dict[str, Any]:
        """Schéma des paramètres d'un modèle."""
        try:
            from datakit.models.models.registry import ModelRegistry
            registry = ModelRegistry()
            return registry.get_parameter_schema(model_name)
        except (ImportError, ValueError):
            # Fallback avec des schémas par défaut
            default_schemas = {
                "RandomForest": {
                    "n_estimators": {"type": "int", "default": 100, "min": 1, "max": 1000},
                    "max_depth": {"type": "int", "default": None, "min": 1, "max": 100},
                    "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 20},
                },
                "XGBoost": {
                    "n_estimators": {"type": "int", "default": 100, "min": 1, "max": 1000},
                    "learning_rate": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0},
                    "max_depth": {"type": "int", "default": 6, "min": 1, "max": 20},
                },
                "LogisticRegression": {
                    "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 100.0},
                    "max_iter": {"type": "int", "default": 100, "min": 10, "max": 1000},
                },
                "KNN": {
                    "n_neighbors": {"type": "int", "default": 5, "min": 1, "max": 50},
                    "weights": {"type": "str", "default": "uniform", "choices": ["uniform", "distance"]},
                },
            }
            return default_schemas.get(model_name, {"n_estimators": {"type": "int", "default": 100}})

    def train(
        self,
        model_name: str,
        task: str,
        params: Dict[str, Any],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Entraîner un modèle.

        Args:
            model_name: Nom du modèle
            task: Type de tâche ('classification' ou 'regression')
            params: Paramètres du modèle
            test_size: Taille du test set
            random_state: Seed aléatoire

        Returns:
            Résultats de l'entraînement
        """
        from datakit.models.models.factory import ModelFactory
        from datakit.preprocessing.utils.target_detection import detect_target_column

        # Récupérer les données (priorité aux données traitées)
        df = data_state.processed_dataframe
        if df is None or df.empty:
            df = data_state.dataframe
        
        if df is None or df.empty:
            raise ValueError("Aucune donnée disponible. Veuillez d'abord uploader des données.")

        # Détection de la cible
        target_col = detect_target_column(df)
        if target_col is None:
            raise ValueError("Aucune colonne cible détectée. Colonnes attendues: 'target', 'y', 'label', 'class'")

        # Séparer X et y
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Créer et entraîner le modèle
        factory = ModelFactory()
        model = factory.create_model(model_name, task=task, **params)

        metrics = model.train(X, y, test_size=test_size, random_state=random_state)
        cv_results = model.cross_validate(X, y, cv=5)

        self._model = model
        self._metrics = metrics

        return {
            "metrics": metrics,
            "cv_results": cv_results
        }

    def get_status(self) -> Dict[str, Any]:
        """Statut du modèle entraîné."""
        return {
            "has_model": self._model is not None,
            "metrics": self._metrics
        }