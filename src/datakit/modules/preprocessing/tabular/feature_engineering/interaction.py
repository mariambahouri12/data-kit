from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ...base import BasePreprocessor
from .._column_utils import select_columns

class InteractionFeatureCreator(BasePreprocessor):
    """Crée des features de produit (col_a * col_b * ...), avec garde-fou sur la taille de sortie."""

    def __init__(self,
                 columns: Optional[List[str]] = None,
                 max_interactions: int = 2,
                 max_output_features: int = 5000,
                 **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.max_interactions = max_interactions
        self.max_output_features = max_output_features
        self.interaction_combinations: List[Tuple[str, ...]] = []

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        cols_to_use = select_columns(X, self.columns, dtype_include=[np.number])

        combos = [
            combo
            for r in range(2, min(self.max_interactions, len(cols_to_use)) + 1)
            for combo in combinations(cols_to_use, r)
        ]

        if len(combos) > self.max_output_features:
            raise ValueError(
                f"Interaction features would create {len(combos)} columns "
                f"(max is {self.max_output_features}). Reduce max_interactions or number of columns."
            )

        self.interaction_combinations = combos

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for combo in self.interaction_combinations:
            col_name = "*".join(combo)
            X_copy[col_name] = X_copy[list(combo)].prod(axis=1)
        return X_copy