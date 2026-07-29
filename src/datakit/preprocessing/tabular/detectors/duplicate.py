# preprocessing/tabular/detectors/duplicate.py

from typing import Optional

import pandas as pd

from ...base import BaseDetector


class DuplicateDetector(BaseDetector):

    """Détecte les lignes dupliquées."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.duplicate_count = 0

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.problems = []
        self.duplicate_count = int(X.duplicated().sum())

        if self.duplicate_count > 0:
            self.problems.append({
                "description": f"{self.duplicate_count} lignes dupliquées",
            })

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X