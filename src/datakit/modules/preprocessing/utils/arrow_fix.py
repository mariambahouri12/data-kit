# preprocessing/utils/arrow_fix.py
"""
Fonctions pour corriger les DataFrames avant l'affichage dans Streamlit.
"""

import pandas as pd
import numpy as np


def fix_dataframe_for_arrow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit un DataFrame pour le rendre compatible avec Arrow.
    
    Args:
        df: DataFrame à corriger
    
    Returns:
        DataFrame corrigé
    """
    if df is None:
        return df
    
    if df.empty:
        return df
    
    df_copy = df.copy()
    
    for col in df_copy.columns:
        try:
            # Convertir StringDtype en object
            if str(df_copy[col].dtype) == 'string':
                df_copy[col] = df_copy[col].astype('object')
                df_copy[col] = df_copy[col].fillna('')
            
            # Convertir category en object
            elif df_copy[col].dtype.name == 'category':
                df_copy[col] = df_copy[col].astype('object')
                df_copy[col] = df_copy[col].fillna('')
            
            # Convertir les colonnes object en string
            elif df_copy[col].dtype == 'object':
                df_copy[col] = df_copy[col].fillna('')
                df_copy[col] = df_copy[col].astype(str)
            
            # Convertir les colonnes avec des types pandas non standard
            elif 'int' in str(df_copy[col].dtype) and not pd.api.types.is_integer_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype('float64')
                df_copy[col] = df_copy[col].fillna(0)
            
            # Convertir les colonnes avec des types pandas non standard
            elif 'float' in str(df_copy[col].dtype) and not pd.api.types.is_float_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype('float64')
                df_copy[col] = df_copy[col].fillna(0.0)
            
        except Exception:
            try:
                df_copy[col] = df_copy[col].astype(str)
                df_copy[col] = df_copy[col].fillna('')
            except:
                pass
    
    return df_copy


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
    
    return fix_dataframe_for_arrow(df)


def fix_dataframe_complete(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correction complète d'un DataFrame pour le rendre Arrow-compatible.
    À utiliser pour les données stockées en session.
    
    Args:
        df: DataFrame à corriger
    
    Returns:
        DataFrame corrigé
    """
    if df is None:
        return df
    
    if df.empty:
        return df
    
    # Appliquer la correction
    df_corrected = fix_dataframe_for_arrow(df)
    
    # Réinitialiser l'index si nécessaire
    if df_corrected.index.dtype == 'object':
        df_corrected = df_corrected.reset_index(drop=True)
    
    return df_corrected


def fix_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alias pour fix_dataframe_complete.
    """
    return fix_dataframe_complete(df)