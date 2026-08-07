"""
Model service - backend interface for datakit.models
"""

from typing import Dict, Any, List

from .state import data_state

from datakit.models.models.factory import ModelFactory
from datakit.models.models.registry import ModelRegistry
from datakit.preprocessing.utils.target_detection import detect_target_column

class ModelService:
    """Service for model training and management."""

    def __init__(self):
        self._model = None
        self._metrics = None

    def get_available_models(self) -> List[str]:
        """
        Return available registered models.
        """
        registry = ModelRegistry()

        return registry.get_available_models()

    def get_parameter_schema(
        self,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Return model parameter schema.
        """

        registry = ModelRegistry()

        return registry.get_parameter_schema(model_name)

    def train(
        self,
        model_name: str,
        task: str,
        params: Dict[str, Any],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train selected model.
        """

        df = (
            data_state.processed_dataframe
            if data_state.processed_dataframe is not None
            else data_state.dataframe
        )


        if df is None or df.empty:
            raise ValueError(
                "No data available. Upload data first."
            )

        target_column = detect_target_column(df)


        if target_column is None:
            raise ValueError(
                "No target column detected."
            )

        X = df.drop(columns=[target_column])
        y = df[target_column]


        factory = ModelFactory()

        model = factory.create_model(
            model_name,
            task=task,
            user_params=params
        )

        metrics = model.train(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

        cv_results = model.cross_validate(
            X,
            y,
            cv=5
        )

        self._model = model
        self._metrics = metrics

        return {
            "metrics": metrics,
            "cv_results": cv_results
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Return current model status.
        """
        return {
            "has_model": self._model is not None,
            "metrics": self._metrics
        }