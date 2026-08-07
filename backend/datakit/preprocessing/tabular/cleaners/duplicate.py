import warnings
from typing import Optional, Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

from ...base import BasePreprocessor
from ..utils._column_utils import select_columns

class DuplicateCleaner(BasePreprocessor):
    """Supprime les lignes dupliquées. Stateless : rien à apprendre au fit."""

    def __init__(self, subset: Optional[List[str]] = None, keep: str = "first", **kwargs):
        super().__init__(**kwargs)
        self.subset = subset
        self.keep = keep

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop_duplicates(subset=self.subset, keep=self.keep)