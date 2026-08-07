from .state import data_state

from datakit.data.adapters import FileAdapter
from datakit.data.loader import FileLoader


class UploadService:


    def __init__(self):
        self.loader = FileLoader()



    def upload(self, file):

        adapted_file = FileAdapter(file)

        df = self.loader.load(adapted_file)

        data_state.dataframe = df

        return {
            "success": True,
            "filename": adapted_file.name
        }