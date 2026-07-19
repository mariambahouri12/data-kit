"""
Chargement des fichiers uploadés (CSV / Excel / Parquet).

Centralise la logique de lecture de fichier, indépendamment de Streamlit,
pour que l'UI n'ait plus à connaître les formats supportés.
"""
from pathlib import Path

import pandas as pd

from datakit.preprocessing.utils.arrow_fix import fix_dataframe_complete


class FileLoader:
    """Lit un fichier uploadé (CSV/Excel/Parquet) et renvoie un DataFrame nettoyé."""

    def __init__(self, sep: str = ",", encoding: str = "utf-8"):
        """
        Args:
            sep: Séparateur CSV
            encoding: Encodage du fichier CSV
        """
        self.sep = sep
        self.encoding = encoding

    def load(self, uploaded_file) -> pd.DataFrame:
        extension = Path(uploaded_file.name).suffix.lower()

        if extension == ".csv":
            df = pd.read_csv(uploaded_file, sep=self.sep, encoding=self.encoding)
        elif extension in (".xlsx", ".xls"):
            df = pd.read_excel(uploaded_file)
        elif extension == ".parquet":
            df = pd.read_parquet(uploaded_file)
        else:
            raise ValueError(f"Format non supporté: {extension}")

        return fix_dataframe_complete(df)