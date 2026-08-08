"""
Preprocessing service.
"""
from typing import Optional, Dict, Any, List

import pandas as pd

from .state import data_state
from .ai_context_state import context_manager

from datakit.preprocessing.tabular.config import PreprocessingConfig
from datakit.preprocessing.presets import PreprocessingPresets
from datakit.preprocessing.factory import PreprocessingFactory
from datakit.preprocessing.tabular.balancers.balance_analyzer import ImbalanceAnalyzer
from datakit.preprocessing.orchestrator import run_preprocessing, run_detection, _json_safe
from datakit.preprocessing.utils.target_detection import detect_target_column


class PreprocessingService:

    def process(
        self,
        config: Dict[str, Any],
        preset: Optional[str] = None,
    ) -> dict:

        if data_state.dataframe is None:
            raise ValueError("No dataset loaded")

        merged_config: Dict[str, Any] = {}
        if preset:
            merged_config.update(PreprocessingPresets.get_preset(preset))
        merged_config.update(config or {})

        preprocess_config = PreprocessingConfig.from_dict(merged_config)

        result = run_preprocessing(
            data_state.dataframe,
            preprocess_config
        )

        data_state.processed_dataframe = result.df

        dataset_name = data_state.filename or "dataset"
        context_manager.update_dataset(result.df, dataset_name=dataset_name)

        for step_name in result.builder.get_step_names():
            context_manager.update_preprocessing(
                operation_name=step_name,
                columns=[],
                parameters={"preset": preset} if preset else None
            )

        if result.balancing_message:
            context_manager.update_preprocessing(
                operation_name="balancing",
                columns=[result.target_column] if result.target_column else [],
                parameters=result.balancing_report or {}
            )

        response: Dict[str, Any] = {
            "success": True,
            "message": "Preprocessing completed",
            "rows": len(result.df),
            "columns": len(result.df.columns),
        }

        if result.balancing_message:
            response["balancing_message"] = result.balancing_message
        if result.balancing_report:
            response["balancing_report"] = result.balancing_report
        if result.target_column:
            response["target_column"] = result.target_column

        response["pipeline_summary"] = result.builder.get_pipeline_summary()
        response["step_details"] = self._extract_step_details(result.pipeline)

        return _json_safe(response)

    @staticmethod
    def _extract_step_details(pipeline) -> Dict[str, Any]:
        if pipeline is None:
            return {}

        details: Dict[str, Any] = {}
        for step_name, step_obj in pipeline.named_steps.items():
            step_info: Dict[str, Any] = {}
            if hasattr(step_obj, "get_feature_names"):
                step_info["feature_names"] = step_obj.get_feature_names()
            if hasattr(step_obj, "get_scale_params"):
                step_info["scale_params"] = step_obj.get_scale_params()
            if hasattr(step_obj, "get_explained_variance"):
                step_info["explained_variance"] = step_obj.get_explained_variance()
            if step_info:
                details[step_name] = step_info
        return details

    def get_processed_dataframe(self) -> Optional[pd.DataFrame]:
        return data_state.processed_dataframe

    def detect_issues(self) -> Dict[str, Any]:
        if data_state.dataframe is None:
            raise ValueError("No dataset loaded")
        return run_detection(data_state.dataframe)

    def suggest_balancing(self) -> Dict[str, Any]:
        if data_state.dataframe is None:
            raise ValueError("No dataset loaded")

        target_col = detect_target_column(data_state.dataframe)
        if target_col is None:
            raise ValueError("No target column found for balance analysis")

        result = ImbalanceAnalyzer.suggest_method(data_state.dataframe[target_col])
        return _json_safe(result)

    @staticmethod
    def list_presets() -> List[str]:
        return PreprocessingPresets.list_presets()

    @staticmethod
    def get_preset(name: str) -> Dict[str, Any]:
        return PreprocessingPresets.get_preset(name)

    @staticmethod
    def list_preprocessors() -> List[Dict[str, Any]]:
        return PreprocessingFactory.list_preprocessors()

    @staticmethod
    def get_preprocessor_info(name: str) -> Dict[str, Any]:
        return PreprocessingFactory.get_preprocessor_info(name)