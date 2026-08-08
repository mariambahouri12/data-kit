
# preprocessing/factory.py

from typing import Dict, Any, Optional, List, Type, Union

import pandas as pd
from sklearn.pipeline import Pipeline

from .base import BasePreprocessor
from .tabular.config import PreprocessingConfig, TaskType
from .tabular.pipeline_builder import PipelineBuilder, SimplePipelineBuilder
from .tabular.detectors import (
    MissingValueDetector,
    OutlierDetector,
    CorrelationDetector,
    CardinalityDetector,
    DuplicateDetector,
)
from .tabular.cleaners import (
    MissingValueCleaner,
    OutlierCleaner,
    DuplicateCleaner,
)
from .tabular.encoders.encoders import CategoricalEncoder
from .tabular.transformers.scalers import FeatureScaler
from .tabular.transformers import (
    LogTransformer,
    SqrtTransformer,
    ReciprocalTransformer,
    BoxCoxTransformer,
    YeoJohnsonTransformer,
    PercentileTransformer,
)
from .tabular.reducers import FeatureSelector, PCAReducer, LDAReducer
from .tabular.balancers.balancers import ClassBalancer
from .tabular.feature_engineering import (
    PolynomialFeatureCreator,
    InteractionFeatureCreator,
    RatioFeatureCreator,
    AggregationFeatureCreator,
    DateFeatureCreator,
)


class PreprocessingFactory:
    """
    Factory for creating preprocessors and pipelines.

    Supports both individual preprocessing components
    and complete preprocessing pipelines.
    """

    # Registry of available preprocessors
    _preprocessors = {
        # Detectors
        "missing_detector": MissingValueDetector,
        "outlier_detector": OutlierDetector,
        "correlation_detector": CorrelationDetector,
        "cardinality_detector": CardinalityDetector,
        "duplicate_detector": DuplicateDetector,

        # Cleaning
        "missing_cleaner": MissingValueCleaner,
        "outlier_cleaner": OutlierCleaner,
        "duplicate_cleaner": DuplicateCleaner,

        # Encoding
        "categorical_encoder": CategoricalEncoder,

        # Scaling
        "feature_scaler": FeatureScaler,

        # Transformations
        "log_transformer": LogTransformer,
        "sqrt_transformer": SqrtTransformer,
        "reciprocal_transformer": ReciprocalTransformer,
        "boxcox_transformer": BoxCoxTransformer,
        "yeojohnson_transformer": YeoJohnsonTransformer,
        "percentile_transformer": PercentileTransformer,

        # Dimensionality reduction
        "feature_selector": FeatureSelector,
        "pca_reducer": PCAReducer,
        "lda_reducer": LDAReducer,

        # Class balancing
        "class_balancer": ClassBalancer,

        # Feature engineering
        "polynomial_creator": PolynomialFeatureCreator,
        "interaction_creator": InteractionFeatureCreator,
        "ratio_creator": RatioFeatureCreator,
        "aggregation_creator": AggregationFeatureCreator,
        "date_creator": DateFeatureCreator,
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> BasePreprocessor:
        """
        Create a preprocessor by name.

        Args:
            name: Name of the preprocessor.
            **kwargs: Preprocessor parameters.

        Returns:
            An instance of the requested preprocessor.
        """
        if name not in cls._preprocessors:
            available = ", ".join(cls._preprocessors.keys())
            raise ValueError(
                f"Preprocessor '{name}' not found. Available: {available}"
            )

        preprocessor_class = cls._preprocessors[name]
        return preprocessor_class(**kwargs)

    @classmethod
    def list_preprocessors(cls) -> List[Dict[str, Any]]:
        """
        List all available preprocessors.

        Returns:
            A list containing information about each preprocessor.
        """
        return [
            {
                "name": name,
                "class": preprocessor_class.__name__,
                "description": (
                    preprocessor_class.__doc__.strip()
                    if preprocessor_class.__doc__
                    else ""
                ),
            }
            for name, preprocessor_class in cls._preprocessors.items()
        ]

    @classmethod
    def create_pipeline(
        cls,
        config: Optional[Union[Dict[str, Any], PreprocessingConfig]] = None,
    ) -> Pipeline:
        """
        Create a complete preprocessing pipeline.

        Args:
            config: Preprocessing configuration, either as a dictionary
                or as a PreprocessingConfig instance.

        Returns:
            An unfitted sklearn Pipeline.

        Note:
            This method previously accepted X and y parameters that were
            never used internally. The pipeline is entirely defined by
            `config`.

            Keeping X and y in the signature would incorrectly suggest
            that they influence pipeline construction, for example through
            automatic column detection.

            The target variable y must instead be provided when calling
            `pipeline.fit()` or `pipeline.fit_transform()` so that
            Target Encoding, CatBoost Encoding, and LDA can receive it.
        """
        if isinstance(config, dict):
            config = PreprocessingConfig.from_dict(config)
        elif config is None:
            config = PreprocessingConfig()

        builder = PipelineBuilder(config)
        return builder.build_pipeline()

    @classmethod
    def create_detection_pipeline(cls) -> Pipeline:
        """Create a preprocessing detection pipeline."""
        builder = PipelineBuilder()
        return builder.build_detection_pipeline()

    @classmethod
    def create_simple_pipeline(
        cls,
        pipeline_type: str = "default",
        task_type: str = "classification",
        **kwargs,
    ) -> Pipeline:
        """
        Create a predefined simple preprocessing pipeline.

        Args:
            pipeline_type:
                Pipeline type: 'default', 'robust',
                'high_performance', or 'minimal'.

            task_type:
                Task type: 'classification' or 'regression'.

            **kwargs:
                Additional preprocessing parameters.

        Returns:
            A configured sklearn Pipeline.
        """
        kwargs["task_type"] = TaskType(task_type)

        if pipeline_type == "default":
            builder = SimplePipelineBuilder.create_default(**kwargs)

        elif pipeline_type == "robust":
            builder = SimplePipelineBuilder.create_robust(**kwargs)

        elif pipeline_type == "high_performance":
            builder = SimplePipelineBuilder.create_high_performance(**kwargs)

        elif pipeline_type == "minimal":
            builder = SimplePipelineBuilder.create_minimal(**kwargs)

        else:
            raise ValueError(
                f"Unknown pipeline type: {pipeline_type}"
            )

        return builder.build_pipeline()

    @classmethod
    def get_preprocessor_info(cls, name: str) -> Dict[str, Any]:
        """
        Get information about a preprocessor.

        Args:
            name: Name of the preprocessor.

        Returns:
            A dictionary containing information about the preprocessor.
        """
        if name not in cls._preprocessors:
            raise ValueError(
                f"Preprocessor '{name}' not found"
            )

        preprocessor_class = cls._preprocessors[name]

        return {
            "name": name,
            "class": preprocessor_class.__name__,
            "description": (
                preprocessor_class.__doc__.strip()
                if preprocessor_class.__doc__
                else ""
            ),
        }

    @classmethod
    def register_preprocessor(
        cls,
        name: str,
        preprocessor_class: Type[BasePreprocessor],
    ):
        """
        Register a new preprocessor.

        Args:
            name: Name under which the preprocessor will be registered.
            preprocessor_class: Preprocessor class to register.
        """
        if not issubclass(preprocessor_class, BasePreprocessor):
            raise ValueError(
                f"{preprocessor_class.__name__} "
                "must inherit from BasePreprocessor"
            )

        cls._preprocessors[name] = preprocessor_class

