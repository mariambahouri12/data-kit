
from typing import Optional

import pandas as pd



TARGET_COLUMN_CANDIDATES = ("target", "y", "label", "class")


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Détecte une colonne cible probable par son nom (heuristique simple)."""
    for col in df.columns:
        if col.lower() in TARGET_COLUMN_CANDIDATES:
            return col
    return None