from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ...base import BasePreprocessor

class AggregationFeatureCreator(BasePreprocessor):
    """Crée des features par agrégation groupée (ex: moyenne d'une colonne par groupe)."""

    DEFAULT_AGGREGATIONS = ("mean", "sum", "std", "min", "max", "count")

    def __init__(self,
                 group_column: Optional[str] = None,
                 agg_columns: Optional[List[str]] = None,
                 aggregations: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.group_column = group_column
        self.agg_columns = agg_columns
        self.aggregations = aggregations or list(self.DEFAULT_AGGREGATIONS)

        self.agg_mapping: Dict[str, Dict[str, Dict]] = {}
        self.agg_names: List[str] = []

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        if self.group_column is None or self.group_column not in X.columns:
            raise ValueError(f"Group column '{self.group_column}' not found")

        agg_columns = self.agg_columns
        if agg_columns is None:
            agg_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            if self.group_column in agg_columns:
                agg_columns.remove(self.group_column)

        # Reset : évite l'accumulation si _fit est appelé plusieurs fois sur le même objet.
        self.agg_mapping = {}
        self.agg_names = []

        for col in agg_columns:
            if col not in X.columns:
                continue
            grouped = X.groupby(self.group_column)[col]
            agg_dict = {agg: grouped.agg(agg).to_dict() for agg in self.aggregations}
            self.agg_mapping[col] = agg_dict
            self.agg_names.extend(f"{col}_{agg}" for agg in self.aggregations)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col, agg_dict in self.agg_mapping.items():
            for agg, mapping in agg_dict.items():
                X_copy[f"{col}_{agg}"] = X_copy[self.group_column].map(mapping)
        return X_copy
