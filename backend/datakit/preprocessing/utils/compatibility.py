import pandas as pd

def _as_dataframe(X, columns) -> pd.DataFrame:
    """Garantit un pd.DataFrame, quelle que soit la version d'imblearn utilisée
    (certaines versions retournent un ndarray brut plutôt qu'un DataFrame)."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=columns)


def _as_series(y, name: str, index) -> pd.Series:
    """Garantit un pd.Series, même si imblearn a retourné un ndarray brut."""
    if isinstance(y, pd.Series):
        return y
    return pd.Series(y, name=name, index=index)