
"""
Shared state for services.
Allows data to be shared between different services.
"""

import pandas as pd
from typing import Optional

class DataState:
    """Global data state for the backend."""

    def __init__(self):
        self._dataframe: Optional[pd.DataFrame] = None
        self._filename: Optional[str] = None
        self._processed_dataframe: Optional[pd.DataFrame] = None

    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        """Returns the loaded DataFrame."""
        return self._dataframe

    @dataframe.setter
    def dataframe(self, df: pd.DataFrame):
        """Sets the loaded DataFrame."""
        self._dataframe = df

    @property
    def filename(self) -> Optional[str]:
        """Returns the name of the loaded file."""
        return self._filename

    @filename.setter
    def filename(self, name: str):
        """Sets the name of the loaded file."""
        self._filename = name

    @property
    def processed_dataframe(self) -> Optional[pd.DataFrame]:
        """Returns the processed DataFrame."""
        return self._processed_dataframe

    @processed_dataframe.setter
    def processed_dataframe(self, df: pd.DataFrame):
        """Sets the processed DataFrame."""
        self._processed_dataframe = df

    def clear(self):
        """Clears all data."""
        self._dataframe = None
        self._filename = None
        self._processed_dataframe = None

    def has_data(self) -> bool:
        """Checks whether data is loaded."""
        return self._dataframe is not None and not self._dataframe.empty

    def has_processed_data(self) -> bool:
        """Checks whether processed data is available."""
        return self._processed_dataframe is not None and not self._processed_dataframe.empty


# Shared singleton instance

data_state = DataState()
