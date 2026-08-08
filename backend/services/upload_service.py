from .state import data_state
from .ai_context_state import context_manager

from datakit.data.adapters import FileAdapter
from datakit.data.loader import FileLoader


class UploadService:

    def __init__(self):
        self.loader = FileLoader()

    def upload(self, file):
        adapted_file = FileAdapter(file)

        df = self.loader.load(adapted_file)

        data_state.dataframe = df
        data_state.filename = adapted_file.name


        context_manager.update_dataset(df, dataset_name=adapted_file.name)

        return {
            "success": True,
            "filename": adapted_file.name
        }