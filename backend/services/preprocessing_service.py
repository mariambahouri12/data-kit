"""
Preprocessing service.
"""

from typing import Optional, Dict, Any

import pandas as pd

from .state import data_state

from datakit.preprocessing.tabular.config import PreprocessingConfig
from datakit.preprocessing.orchestrator import run_preprocessing

class PreprocessingService:

    def process(
        self,
        config: Dict[str, Any]
    ) -> dict:

        if data_state.dataframe is None:
            raise ValueError(
                "No dataset loaded"
            )

        preprocess_config = PreprocessingConfig(
            **config
        )

        result = run_preprocessing(
            data_state.dataframe,
            preprocess_config
        )

        data_state.processed_dataframe = result.df

        return {
            "success": True,
            "message": "Preprocessing completed",
            "rows": len(result.df),
            "columns": len(result.df.columns)
        }

    def get_processed_dataframe(
        self
    ) -> Optional[pd.DataFrame]:

        return data_state.processed_dataframe