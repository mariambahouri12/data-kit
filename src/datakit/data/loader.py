from pathlib import Path

import pandas as pd

from ..exceptions import EmptyFileError


class FileLoader:

    def __init__(self, sep: str = ",", encoding: str = "utf-8"):
        self.sep = sep
        self.encoding = encoding

    def load(self, uploaded_file) -> pd.DataFrame:

        extension = Path(uploaded_file.name).suffix.lower()

        try:
            if extension == ".csv":
                df = pd.read_csv(
                    uploaded_file,
                    sep=self.sep,
                    encoding=self.encoding
                )

            elif extension in (".xlsx", ".xls"):
                df = pd.read_excel(uploaded_file)

            elif extension == ".parquet":
                df = pd.read_parquet(uploaded_file)

            else:
                raise ValueError(
                    f"Unsupported file format: {extension}"
                )

        except pd.errors.EmptyDataError as error:
            raise EmptyFileError(
                "Uploaded file is empty."
            ) from error

        if df.empty:
            raise EmptyFileError(
                "Uploaded file is empty."
            )

        return df