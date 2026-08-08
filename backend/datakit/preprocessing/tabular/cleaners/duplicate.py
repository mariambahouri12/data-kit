
from typing import Optional, List

import pandas as pd


from ...base import BasePreprocessor
from ..utils._column_utils import select_columns

class DuplicateCleaner(BasePreprocessor):
    """Remove duplicated rows. Stateless: nothing to learn during fit."""

    def __init__(self, subset: Optional[List[str]] = None, keep: str = "first", **kwargs):
        super().__init__(**kwargs)
        self.subset = subset
        self.keep = keep

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop_duplicates(subset=self.subset, keep=self.keep)