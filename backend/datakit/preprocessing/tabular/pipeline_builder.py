
# preprocessing/tabular/pipeline_builder.py

"""
Builds a scikit-learn pipeline from a PreprocessingConfig.

Note: class balancing is NOT included in the scikit-learn pipeline — it
changes the number of rows in X and y simultaneously, which the Pipeline
API cannot properly express. Use apply_balancing() separately, before
model training.

Known limitation: the same preprocessing strategy is applied to all
affected columns (no per-column strategy). A future improvement could
use ColumnTransformer if differentiated rules are needed.
"""

from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

from .config import (
    PreprocessingConfig,
    EncodingMethod,
    ScalingMethod,
    ImputationMethod,
    OutlierMethod,
    BalancingMethod,
    FeatureSelectionMethod,
)
from .detectors import (
    MissingValueDetector,
    OutlierDetector,
    CorrelationDetector,
    CardinalityDetector,
    DuplicateDetector,
)
from .cleaners import MissingValueCleaner, OutlierCleaner, DuplicateCleaner
from .encoders.encoders import CategoricalEncoder
from .transformers.scalers import FeatureScaler
from .transformers import (
    LogTransformer,
    BoxCoxTransformer,
    YeoJohnsonTransformer,
)
from .reducers import FeatureSelector, PCAReducer, LDAReducer
from .balancers.balancers import ClassBalancer
from .feature_engineering import (
    PolynomialFeatureCreator,
    InteractionFeatureCreator,
    RatioFeatureCreator,
)

from ..presets import PreprocessingPresets


PipelineStep = Tuple[str, Any]


class PipelineBuilder:
    """Builds a scikit-learn preprocessing pipeline from a configuration."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.detectors = self._build_detectors()

        self.last_balancer: Optional[ClassBalancer] = None

    @staticmethod
    def _build_detectors() -> List[Any]:
        return [
            MissingValueDetector(threshold=0.05),
            OutlierDetector(method="iqr", threshold=1.5),
            CorrelationDetector(threshold=0.8),
            CardinalityDetector(max_categories=50),
            DuplicateDetector(),
        ]

    # -- Main pipeline construction --------------------------------------

    def build_pipeline(self) -> Pipeline:
        """Builds the complete pipeline (excluding balancing, see apply_balancing)."""
        step_builders = (
            self._step_drop_duplicates,
            self._step_drop_high_missing,
            self._step_imputation,
            self._step_outlier_handling,
            self._step_polynomial,
            self._step_interactions,
            self._step_ratios,
            self._step_log_transform,
            self._step_boxcox,
            self._step_yeojohnson,
            self._step_encoding,
            self._step_scaling,
            self._step_feature_selection,
            self._step_reduction,
        )

        steps: List[PipelineStep] = []

        for build_step in step_builders:
            step = build_step()

            if step is not None:
                steps.append(step)

        return Pipeline(steps)

    # -- One method per step, each returns None if the step is disabled --

    def _step_drop_duplicates(self) -> Optional[PipelineStep]:
        if not self.config.drop_duplicates:
            return None

        return ("drop_duplicates", DuplicateCleaner())

    def _step_drop_high_missing(self) -> Optional[PipelineStep]:
        if not self.config.drop_high_missing:
            return None

        return (
            "drop_high_missing",
            self._make_drop_high_missing_transformer()
        )

    def _step_imputation(self) -> Optional[PipelineStep]:
        method = self.config.imputation_method

        if method == ImputationMethod.DROP:
            return None

        imputer = MissingValueCleaner(
            strategy=method,
            fill_value=self.config.imputation_fill_value,
            columns=self.config.imputation_columns,
            knn_neighbors=self.config.imputation_knn_neighbors,
        )

        return ("imputation", imputer)

    def _step_outlier_handling(self) -> Optional[PipelineStep]:
        method = self.config.outlier_method

        if method == OutlierMethod.NONE:
            return None

        cleaner = OutlierCleaner(
            method=method,
            threshold=self.config.outlier_threshold,
            action=self.config.outlier_action,
            columns=self.config.outlier_columns,
        )

        return ("outlier_handling", cleaner)

    def _step_polynomial(self) -> Optional[PipelineStep]:
        if not self.config.create_polynomial:
            return None

        creator = PolynomialFeatureCreator(
            degree=self.config.polynomial_degree,
            max_features=self.config.polynomial_max_features,
            max_output_features=self.config.polynomial_max_output_features,
        )

        return ("polynomial", creator)

    def _step_interactions(self) -> Optional[PipelineStep]:
        if not self.config.create_interactions:
            return None

        return ("interactions", InteractionFeatureCreator())

    def _step_ratios(self) -> Optional[PipelineStep]:
        if not self.config.create_ratios:
            return None

        creator = RatioFeatureCreator(
            max_pairs=self.config.ratios_max_pairs
        )

        return ("ratios", creator)

    def _step_log_transform(self) -> Optional[PipelineStep]:
        if not self.config.apply_log_transform:
            return None

        return (
            "log_transform",
            LogTransformer(columns=self.config.transform_columns)
        )

    def _step_boxcox(self) -> Optional[PipelineStep]:
        if not self.config.apply_boxcox:
            return None

        transformer = BoxCoxTransformer(
            columns=self.config.transform_columns,
            lambda_=self.config.transform_lambda,
        )

        return ("boxcox", transformer)

    def _step_yeojohnson(self) -> Optional[PipelineStep]:
        if not self.config.apply_yeojohnson:
            return None

        transformer = YeoJohnsonTransformer(
            columns=self.config.transform_columns,
            lambda_=self.config.transform_lambda,
        )

        return ("yeojohnson", transformer)

    def _step_encoding(self) -> Optional[PipelineStep]:
        method = self.config.encoding_method

        if method == EncodingMethod.NONE:
            return None

        encoder = CategoricalEncoder(
            method=method,
            columns=self.config.encoding_columns,
            max_categories=self.config.encoding_max_categories,
            min_frequency=self.config.encoding_min_frequency,
            handle_unknown=self.config.encoding_handle_unknown,
            sparse=self.config.encoding_sparse,
        )

        return ("encoding", encoder)

    def _step_scaling(self) -> Optional[PipelineStep]:
        method = self.config.scaling_method

        if method == ScalingMethod.NONE:
            return None

        scaler = FeatureScaler(
            method=method,
            columns=self.config.scaling_columns,
            with_mean=self.config.scaling_with_mean,
            with_std=self.config.scaling_with_std,
        )

        return ("scaling", scaler)

    def _step_feature_selection(self) -> Optional[PipelineStep]:
        method = self.config.feature_selection_method

        if method == FeatureSelectionMethod.NONE:
            return None

        selector = FeatureSelector(
            method=method,
            threshold=self.config.feature_selection_threshold,
            k=self.config.feature_selection_k,
            columns=self.config.feature_selection_columns,
            task_type=self.config.task_type,
        )

        return ("feature_selection", selector)

    def _step_reduction(self) -> Optional[PipelineStep]:
        if self.config.reduction_method == "pca":
            reducer = PCAReducer(
                n_components=self.config.reduction_components,
                variance_ratio=self.config.reduction_variance_ratio,
            )

            return ("pca", reducer)

        if self.config.reduction_method == "lda":
            reducer = LDAReducer(
                n_components=self.config.reduction_components
            )

            return ("lda", reducer)

        return None

    def _make_drop_high_missing_transformer(self) -> FunctionTransformer:
        threshold = self.config.high_missing_threshold
        verbose = self.config.verbose

        def drop_high_missing(X: pd.DataFrame) -> pd.DataFrame:
            missing_pct = X.isnull().mean()
            cols_to_drop = missing_pct[
                missing_pct > threshold
            ].index.tolist()

            if cols_to_drop and verbose:
                print(
                    f"Dropping columns with > {threshold * 100}% "
                    f"missing: {cols_to_drop}"
                )

            return (
                X.drop(columns=cols_to_drop)
                if cols_to_drop
                else X
            )

        return FunctionTransformer(drop_high_missing)

    # -- Balancing (outside the scikit-learn pipeline) -------------------

    def apply_balancing(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply separately, before model training.

        The only method in the codebase that builds a ClassBalancer from
        the configuration — orchestrator.apply_balancing_if_needed()
        delegates here instead of duplicating this construction.
        """
        method = self.config.balancing_method

        if method == BalancingMethod.NONE:
            self.last_balancer = None
            return X, y

        balancer = ClassBalancer(
            method=method,
            sampling_strategy=self.config.balancing_sampling_strategy,
            random_state=self.config.balancing_random_state,
        )

        X_resampled, y_resampled = balancer.fit_resample(X, y)

        self.last_balancer = balancer

        return X_resampled, y_resampled

    # -- Detection & introspection --------------------------------------

    def build_detection_pipeline(self) -> Pipeline:
        return make_pipeline(*self.detectors)

    def get_step_names(self) -> List[str]:
        return self.config.get_active_steps()

    def get_pipeline_summary(self) -> Dict[str, Any]:
        return {
            "steps": self.get_step_names(),
            "n_steps": len(self.get_step_names()),
            "config": self.config.to_dict(),
        }


class SimplePipelineBuilder(PipelineBuilder):
    """
    Pipeline builder with ready-to-use configurations.
    """

    # pipeline_type (internal, historical API) -> preset name in presets.py
    _PRESET_MAP = {
        "default": "quick",
        "robust": "robust",
        "high_performance": "high_performance",
        "minimal": "minimal",
    }

    # Additional parameters specific to SimplePipelineBuilder, absent from
    # the shared presets (do not add them to presets.py: they would change
    # the preset behavior for all API consumers).
    _EXTRA_DEFAULTS: Dict[str, Dict[str, Any]] = {
        "default": {"encoding_sparse": False},
        "high_performance": {"polynomial_max_features": 20},
    }

    def __init__(self, **kwargs):
        super().__init__(PreprocessingConfig(**kwargs))

    @classmethod
    def _from_preset(
        cls,
        pipeline_type: str,
        **kwargs
    ) -> "SimplePipelineBuilder":
        preset_name = cls._PRESET_MAP[pipeline_type]

        params: Dict[str, Any] = dict(
            PreprocessingPresets.get_preset(preset_name)
        )

        params.update(
            cls._EXTRA_DEFAULTS.get(pipeline_type, {})
        )

        params.update(kwargs)  # explicit kwargs always take precedence

        return cls(**params)

    @classmethod
    def create_default(cls, **kwargs) -> "SimplePipelineBuilder":
        return cls._from_preset("default", **kwargs)

    @classmethod
    def create_robust(cls, **kwargs) -> "SimplePipelineBuilder":
        return cls._from_preset("robust", **kwargs)

    @classmethod
    def create_high_performance(cls, **kwargs) -> "SimplePipelineBuilder":
        return cls._from_preset("high_performance", **kwargs)

    @classmethod
    def create_minimal(cls, **kwargs) -> "SimplePipelineBuilder":
        return cls._from_preset("minimal", **kwargs)

