"""
État partagé pour les services.
Permet de partager les données entre les différents services.
"""

import pandas as pd
from typing import Optional


class DataState:
    """État global des données pour le backend."""
    
    def __init__(self):
        self._dataframe: Optional[pd.DataFrame] = None
        self._filename: Optional[str] = None
        self._processed_dataframe: Optional[pd.DataFrame] = None
    
    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        """Retourne le DataFrame chargé."""
        return self._dataframe
    
    @dataframe.setter
    def dataframe(self, df: pd.DataFrame):
        """Définit le DataFrame chargé."""
        self._dataframe = df
    
    @property
    def filename(self) -> Optional[str]:
        """Retourne le nom du fichier chargé."""
        return self._filename
    
    @filename.setter
    def filename(self, name: str):
        """Définit le nom du fichier chargé."""
        self._filename = name
    
    @property
    def processed_dataframe(self) -> Optional[pd.DataFrame]:
        """Retourne le DataFrame traité."""
        return self._processed_dataframe
    
    @processed_dataframe.setter
    def processed_dataframe(self, df: pd.DataFrame):
        """Définit le DataFrame traité."""
        self._processed_dataframe = df
    
    def clear(self):
        """Efface toutes les données."""
        self._dataframe = None
        self._filename = None
        self._processed_dataframe = None
    
    def has_data(self) -> bool:
        """Vérifie si des données sont chargées."""
        return self._dataframe is not None and not self._dataframe.empty
    
    def has_processed_data(self) -> bool:
        """Vérifie si des données traitées sont disponibles."""
        return self._processed_dataframe is not None and not self._processed_dataframe.empty


# Instance unique partagée
data_state = DataState()