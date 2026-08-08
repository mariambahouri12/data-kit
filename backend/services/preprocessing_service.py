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
from datakit.preprocessing.tabular.balancers.balance_analyzer import (
    ImbalanceAnalyzer
)
from datakit.preprocessing.orchestrator import (
    run_preprocessing,
    run_detection,
    _json_safe,
)
from datakit.preprocessing.utils.target_detection import (
    detect_target_column,
)


class PreprocessingService:
    """
    Service responsible for dataset preprocessing.

    The original dataframe is kept in ``data_state.dataframe``.
    The transformed dataframe is stored in
    ``data_state.processed_dataframe``.

    The AI dataset-quality context is generated from the ORIGINAL
    dataframe when the dataset is uploaded. It is therefore not
    regenerated from the processed dataframe.
    """

    def process(
        self,
        config: Dict[str, Any],
        preset: Optional[str] = None,
    ) -> dict:
        """
        Apply preprocessing to the currently loaded dataset.

        The preprocessing pipeline operates on the original dataframe
        and stores the resulting dataframe separately.

  
        """

        # =========================================================
        # Validate dataset
        # =========================================================

        if data_state.dataframe is None:
            raise ValueError("No dataset loaded")

        # =========================================================
        # Build preprocessing configuration
        # =========================================================

        merged_config: Dict[str, Any] = {}

        if preset:
            merged_config.update(
                PreprocessingPresets.get_preset(preset)
            )

        merged_config.update(config or {})

        preprocess_config = PreprocessingConfig.from_dict(
            merged_config
        )

        # =========================================================
        # Run preprocessing
        # =========================================================

        result = run_preprocessing(
            data_state.dataframe,
            preprocess_config
        )

        # =========================================================
        # Store processed dataframe
        # =========================================================

        data_state.processed_dataframe = result.df

        # =========================================================
        # Store preprocessing history
        # =========================================================

        for step_name in result.builder.get_step_names():

            context_manager.update_preprocessing(
                operation_name=step_name,
                columns=[],
                parameters={
                    "preset": preset
                } if preset else None,
            )

        # =========================================================
        # Store balancing operation
        # =========================================================

        if result.balancing_message:

            context_manager.update_preprocessing(
                operation_name="balancing",
                columns=(
                    [result.target_column]
                    if result.target_column
                    else []
                ),
                parameters=result.balancing_report or {},
            )

        # =========================================================
        # Build response
        # =========================================================

        response: Dict[str, Any] = {
            "success": True,
            "message": "Preprocessing completed",
            "rows": len(result.df),
            "columns": len(result.df.columns),
        }

        # =========================================================
        # Balancing information
        # =========================================================

        if result.balancing_message:
            response["balancing_message"] = (
                result.balancing_message
            )

        if result.balancing_report:
            response["balancing_report"] = (
                result.balancing_report
            )

        # =========================================================
        # Target information
        # =========================================================

        if result.target_column:
            response["target_column"] = (
                result.target_column
            )

        # =========================================================
        # Pipeline information
        # =========================================================

        response["pipeline_summary"] = (
            result.builder.get_pipeline_summary()
        )

        response["step_details"] = (
            self._extract_step_details(
                result.pipeline
            )
        )

        return _json_safe(response)

    # =============================================================
    # Pipeline details
    # =============================================================

    @staticmethod
    def _extract_step_details(
        pipeline
    ) -> Dict[str, Any]:
        """
        Extract useful information from preprocessing steps.
        """

        if pipeline is None:
            return {}

        details: Dict[str, Any] = {}

        for step_name, step_obj in pipeline.named_steps.items():

            step_info: Dict[str, Any] = {}

            # -----------------------------------------------------
            # Feature names
            # -----------------------------------------------------

            if hasattr(step_obj, "get_feature_names"):
                step_info["feature_names"] = (
                    step_obj.get_feature_names()
                )

            # -----------------------------------------------------
            # Scaling parameters
            # -----------------------------------------------------

            if hasattr(step_obj, "get_scale_params"):
                step_info["scale_params"] = (
                    step_obj.get_scale_params()
                )

            # -----------------------------------------------------
            # Explained variance
            # -----------------------------------------------------

            if hasattr(
                step_obj,
                "get_explained_variance"
            ):
                step_info["explained_variance"] = (
                    step_obj.get_explained_variance()
                )

            # -----------------------------------------------------
            # Store step information
            # -----------------------------------------------------

            if step_info:
                details[step_name] = step_info

        return details

    # =============================================================
    # Processed dataframe
    # =============================================================

    def get_processed_dataframe(
        self
    ) -> Optional[pd.DataFrame]:
        """
        Return the processed dataframe.
        """

        return data_state.processed_dataframe

    # =============================================================
    # Detection
    # =============================================================

    def detect_issues(self) -> Dict[str, Any]:
        """
        Detect data-quality issues on the ORIGINAL dataframe.

        Detection is intentionally performed on the original dataset,
        not on the processed dataframe.
        """

        if data_state.dataframe is None:
            raise ValueError("No dataset loaded")

        return run_detection(
            data_state.dataframe
        )

    # =============================================================
    # Balancing suggestion
    # =============================================================

    def suggest_balancing(self) -> Dict[str, Any]:
        """
        Suggest a class-balancing strategy based on the target column.
        """

        if data_state.dataframe is None:
            raise ValueError("No dataset loaded")

        target_col = detect_target_column(
            data_state.dataframe
        )

        if target_col is None:
            raise ValueError(
                "No target column found for balance analysis"
            )

        result = ImbalanceAnalyzer.suggest_method(
            data_state.dataframe[target_col]
        )

        return _json_safe(result)

    # =============================================================
    # Presets
    # =============================================================

    @staticmethod
    def list_presets() -> List[str]:
        """
        Return available preprocessing presets.
        """

        return PreprocessingPresets.list_presets()

    @staticmethod
    def get_preset(
        name: str
    ) -> Dict[str, Any]:
        """
        Return a preprocessing preset configuration.
        """

        return PreprocessingPresets.get_preset(
            name
        )

    # =============================================================
    # Preprocessors
    # =============================================================

    @staticmethod
    def list_preprocessors() -> List[Dict[str, Any]]:
        """
        Return available preprocessing operations.
        """

        return PreprocessingFactory.list_preprocessors()

    @staticmethod
    def get_preprocessor_info(
        name: str
    ) -> Dict[str, Any]:
        """
        Return information about a preprocessing operation.
        """

        return PreprocessingFactory.get_preprocessor_info(
            name
        )