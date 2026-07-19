# preprocessing/tabular/detectors/duplicate.py

from typing import Optional

import pandas as pd

from ...base import BaseDetector


class DuplicateDetector(BaseDetector):

    """Détecte les lignes dupliquées."""

    HIGH_SEVERITY_THRESHOLD = 100

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.duplicate_count = 0

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.problems = []
        self.duplicate_count = int(X.duplicated().sum())

        if self.duplicate_count > 0:
            severity = "medium" if self.duplicate_count > self.HIGH_SEVERITY_THRESHOLD else "low"
            self.problems.append({
                "description": f"{self.duplicate_count} lignes dupliquées",
                "severity": severity,
                "suggestion": "Supprimer les doublons",
            })

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X