# preprocessing/tabular/detectors/correlation.py

import warnings
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from ...base import BaseDetector


class CorrelationDetector(BaseDetector):

    """Détecte les paires de colonnes numériques fortement corrélées."""

    def __init__(self, threshold: float = 0.8, **kwargs):
        """
        Args:
            threshold: Seuil de corrélation absolue (défaut : 0.8).
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.correlations: Dict[str, Any] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.problems = []
        self.correlations = {}

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return

        data = X[numeric_cols].dropna()
        if len(data) < 2:
            warnings.warn("Not enough data after dropping NaNs for correlation", RuntimeWarning)
            return

        corr_matrix = data.corr().abs()
        high_corr_pairs = self._find_high_corr_pairs(corr_matrix, numeric_cols)

        self.correlations = {"matrix": corr_matrix, "high_corr_pairs": high_corr_pairs}

        for pair in high_corr_pairs:
            self.problems.append({
                "description": (
                    f"Corrélation élevée entre {pair['col1']} et {pair['col2']}: "
                    f"{pair['correlation']:.2f}"
                ),
                "severity": "medium",
                "suggestion": "Supprimer une des deux colonnes ou utiliser PCA",
            })

    def _find_high_corr_pairs(self, corr_matrix: pd.DataFrame, columns) -> List[Dict[str, Any]]:
        pairs = []
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                corr_value = corr_matrix.iloc[i, j]
                if corr_value > self.threshold:
                    pairs.append({"col1": columns[i], "col2": columns[j], "correlation": corr_value})
        return pairs

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class CardinalityDetector(BaseDetector):
    """Détecte les colonnes catégorielles à cardinalité trop élevée."""

    HIGH_SEVERITY_THRESHOLD = 100

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
                severity = "high" if n_unique > self.HIGH_SEVERITY_THRESHOLD else "medium"
                self.problems.append({
                    "column": col,
                    "description": f"{n_unique} catégories (recommandé: < {self.max_categories})",
                    "severity": severity,
                    "suggestion": self._suggest_encoding(n_unique),
                })

    @staticmethod
    def _suggest_encoding(n_unique: int) -> str:
        if n_unique < 10:
            return "One-Hot Encoding"
        if n_unique < 50:
            return "Target Encoding"
        return "Frequency Encoding ou Binary Encoding"

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X
