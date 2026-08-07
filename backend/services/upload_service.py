"""
Upload service - gestion des fichiers uploadés.
"""

import io
import traceback
from typing import Optional

import pandas as pd

from datakit.data.loader import FileLoader
from .state import data_state


class UploadService:
    """
    Service pour l'upload et la gestion des données.
    """

    def __init__(self):
        self._file_loader = FileLoader()


    def upload(self, file) -> dict:
        """
        Uploader un fichier.

        Args:
            file: Fichier uploadé (FastAPI UploadFile)

        Returns:
            Informations sur les données chargées
        """

        try:
            # Lire le contenu du fichier
            content = file.file.read()
            filename = file.filename or "unknown"


            class FileWrapper:
                """
                Wrapper compatible pandas.
                Convertit les bytes FastAPI en fichier texte.
                """

                def __init__(self, name: str, content: bytes):
                    self.name = name

                    try:
                        text = content.decode("utf-8")
                    except UnicodeDecodeError:
                        text = content.decode("latin-1")

                    self._io = io.StringIO(text)
                    self._closed = False


                def read(self, size: int = -1) -> str:
                    if self._closed:
                        return ""
                    return self._io.read(size)


                def readline(self, size: int = -1) -> str:
                    if self._closed:
                        return ""
                    return self._io.readline(size)


                def readlines(self, hint: int = -1):
                    if self._closed:
                        return []
                    return self._io.readlines(hint)


                def seek(self, offset: int, whence: int = 0):
                    if self._closed:
                        return 0
                    return self._io.seek(offset, whence)


                def tell(self):
                    if self._closed:
                        return 0
                    return self._io.tell()


                def close(self):
                    self._closed = True
                    self._io.close()


                def __iter__(self):
                    self.seek(0)
                    return self


                def __next__(self):
                    line = self._io.readline()

                    if line == "":
                        raise StopIteration

                    return line


                def __enter__(self):
                    return self


                def __exit__(self, exc_type, exc_val, exc_tb):
                    self.close()


                @property
                def closed(self):
                    return self._closed



            # Création du fichier compatible pandas
            wrapped_file = FileWrapper(
                filename,
                content
            )


            # Chargement avec FileLoader
            df = self._file_loader.load(
                wrapped_file
            )


            # Sauvegarde dans l'état partagé
            data_state.dataframe = df
            data_state.filename = filename

            # Réinitialiser les anciennes données preprocessées
            data_state.processed_dataframe = None


            return {
                "success": True,
                "filename": filename,
                "rows": len(df),
                "columns": len(df.columns),
                "columns_list": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "preview": df.head(5).to_dict(
                    orient="records"
                )
            }


        except Exception as e:

            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }



    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Récupérer les données chargées.
        """

        return data_state.dataframe



    def get_preview(self, limit: int = 100) -> Optional[dict]:
        """
        Récupérer un aperçu des données.
        """

        df = data_state.dataframe

        if df is None or df.empty:
            return None


        return {
            "rows": len(df),
            "columns": len(df.columns),
            "columns_list": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": df.isnull().sum().to_dict(),
            "missing_percent": (
                df.isnull().sum() / len(df) * 100
            ).round(2).to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "memory_usage_mb": round(
                df.memory_usage(deep=True).sum()
                /
                (1024 * 1024),
                2
            ),
            "preview": df.head(limit).to_dict(
                orient="records"
            )
        }



    def clear(self):
        """
        Effacer les données.
        """

        data_state.clear()