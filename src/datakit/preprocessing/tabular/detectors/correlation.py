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

