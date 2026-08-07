"""
Preprocessing service - utilise datakit.preprocessing.
"""

import pandas as pd
from typing import Optional, Dict, Any

from .state import data_state


class PreprocessingService:
    """Service pour le preprocessing."""

    def __init__(self):
        pass

    def process(self, config: Dict[str, Any]) -> dict:
        """
        Appliquer le preprocessing.

        Args:
            config: Configuration du preprocessing

        Returns:
            Résultat du preprocessing
        """
        df = data_state.dataframe
        if df is None or df.empty:
            return {
                "success": False,
                "error": "Aucune donnée chargée. Veuillez d'abord uploader un fichier."
            }

        try:
            from datakit.preprocessing.tabular.config import PreprocessingConfig
            from datakit.preprocessing.orchestrator import run_preprocessing

            # Convertir la config
            preprocess_config = PreprocessingConfig(**config)
            
            # Exécuter le preprocessing
            result = run_preprocessing(df, preprocess_config)

            # Stocker les données traitées
            data_state.processed_dataframe = result.df

            return {
                "success": True,
                "rows": len(result.df),
                "columns": len(result.df.columns),
                "columns_list": result.df.columns.tolist(),
                "message": result.balancing_message or "Preprocessing terminé avec succès",
                "preview": result.df.head(5).to_dict(orient="records")
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_processed(self) -> Optional[dict]:
        """Récupérer les données traitées."""
        df = data_state.processed_dataframe
        if df is None:
            return None

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "columns_list": df.columns.tolist(),
            "preview": df.head(100).to_dict(orient="records")
        }

    def get_processed_dataframe(self) -> Optional[pd.DataFrame]:
        """Récupérer le DataFrame traité."""
        return data_state.processed_dataframe