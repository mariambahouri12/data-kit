# preprocessing/tabular/detectors/cardinality.py

from typing import Dict, Optional

import pandas as pd

from ...base import BaseDetector


class CardinalityDetector(BaseDetector):

    """Détecte les colonnes catégorielles à cardinalité trop élevée."""

    def __init__(self, max_categories: int = 50, **kwargs):
        """
        Args:
            max_categories: Nombre maximum de catégories recommandé.
        """
        super().__init__(**kwargs)
        self.max_categories = max_categories
        self.cardinality_stats: Dict[str, int] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.problems = []
        self.cardinality_stats = {}

        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            n_unique = X[col].nunique()
            self.cardinality_stats[col] = n_unique

            if n_unique > self.max_categories:
                self.problems.append({
                    "column": col,
                    "description": f"{n_unique} catégories (recommandé: < {self.max_categories})",
                })

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X