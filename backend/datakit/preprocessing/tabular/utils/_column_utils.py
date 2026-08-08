
# _column_utils.py

from typing import List, Optional

import pandas as pd


def select_columns(
    X: pd.DataFrame,
    requested: Optional[List[str]],
    dtype_include: Optional[List[str]] = None,
) -> List[str]:
    """
    Resolves the list of columns to process.

    - If `requested` is None: all columns of X (filtered by dtype if provided).
    - Otherwise: the intersection between `requested` and the columns actually
      present in X (silently ignores missing columns).
    """
    if requested is None:
        candidates = X.columns

        if dtype_include is not None:
            candidates = X.select_dtypes(include=dtype_include).columns

        return candidates.tolist()

    return [c for c in requested if c in X.columns]

