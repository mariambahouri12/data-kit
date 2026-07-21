"""
Loading uploaded files (CSV / Excel / Parquet).

"""
from pathlib import Path

import pandas as pd


class FileLoader:
    """Reads an uploaded file (CSV, Excel, or Parquet) and returns a DataFrame."""

    def __init__(self, sep: str = ",", encoding: str = "utf-8"):
        """
        Args:
            sep: CSV delimiter.
            encoding: CSV file encoding.
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
            raise ValueError(f"Unsupported file format: {extension}")

        return df