from typing import List, Optional

import pandas as pd


def select_columns(
    X: pd.DataFrame,
    requested: Optional[List[str]],
    dtype_include: Optional[List[str]] = None,
) -> List[str]:
    """
    Résout la liste de colonnes à traiter.

    - Si `requested` est None : toutes les colonnes de X (filtrées par dtype si fourni).
    - Sinon : l'intersection entre `requested` et les colonnes réellement présentes
      dans X (silencieusement ignore celles qui manquent).
    """
    if requested is None:
        candidates = X.columns
        if dtype_include is not None:
            candidates = X.select_dtypes(include=dtype_include).columns
        return candidates.tolist()

    return [c for c in requested if c in X.columns]