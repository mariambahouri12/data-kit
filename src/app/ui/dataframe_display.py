
import pandas as pd




def safe_display_dataframe(df: pd.DataFrame, max_rows: int = 100) -> pd.DataFrame:
    """
    Préparer un DataFrame pour l'affichage sécurisé dans Streamlit.
    
    Args:
        df: DataFrame à afficher
        max_rows: Nombre maximum de lignes
    
    Returns:
        DataFrame prêt pour l'affichage
    """
    if df is None:
        return df
    
    if df.empty:
        return df

    if len(df) > max_rows:
        df = df.head(max_rows)
    
    return df