# preprocessing/tabular/detectors/cardinality.py

from typing import Dict, Optional

import pandas as pd

from ...base import BaseDetector


class CardinalityDetector(BaseDetector):

    """Detect columns with too high cardinality."""

    def __init__(self, max_categories: int = 50, **kwargs):
        """
        Args:
            max_categories: Maximum recommended number of categories.
        """
        super().__init__(**kwargs)
        self.max_categories = max_categories
        self.cardinality_stats: Dict[str, int] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:

        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            n_unique = X[col].nunique()
            self.cardinality_stats[col] = n_unique

            if n_unique > self.max_categories:
                self.problems.append({
                    "column": col,
                    "description": f"{n_unique} categories (recommanded: < {self.max_categories})",
                })

