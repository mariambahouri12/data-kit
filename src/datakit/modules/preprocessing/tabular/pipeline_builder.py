# preprocessing/tabular/pipeline_builder.py
"""
Construit un pipeline sklearn à partir d'une PreprocessingConfig.

Note : le rééquilibrage de classes (balancing) n'est PAS inclus dans le
pipeline sklearn — il change le nombre de lignes de X et y simultanément,
ce que l'API Pipeline ne permet pas d'exprimer. Utiliser apply_balancing()
séparément, avant l'entraînement du modèle.

Limite connue : une même stratégie de prétraitement est appliquée à toutes
les colonnes concernées (pas de stratégie par colonne). Amélioration future
possible via ColumnTransformer si besoin de règles différenciées.
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
    MissingValueDetector, OutlierDetector, CorrelationDetector,
    CardinalityDetector, DuplicateDetector,
)
from .cleaners import MissingValueCleaner, OutlierCleaner, DuplicateCleaner
from .encoders.encoders import CategoricalEncoder
from .scalers import FeatureScaler
from .transformers import boxcox, log, yeojohnson
from .reducers import FeatureSelector, PCAReducer, LDAReducer
from .balancers import ClassBalancer
from .feature_engineering import (
    PolynomialFeatureCreator, InteractionFeatureCreator, RatioFeatureCreator,
)

PipelineStep = Tuple[str, Any]


class PipelineBuilder:
    """Construit un pipeline de prétraitement sklearn à partir d'une config."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.detectors = self._build_detectors()

    @staticmethod
    def _build_detectors() -> List[Any]:
        return [
            MissingValueDetector(threshold=0.05),
            OutlierDetector(method="iqr", threshold=1.5),
            CorrelationDetector(threshold=0.8),
            CardinalityDetector(max_categories=50),
            DuplicateDetector(),
        ]

    # -- Construction du pipeline principal ---------------------------------

    def build_pipeline(self) -> Pipeline:
        """Construit le pipeline complet (hors balancing, voir apply_balancing)."""
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

    # -- Une méthode par étape, chacune retourne None si l'étape est désactivée --

    def _step_drop_duplicates(self) -> Optional[PipelineStep]:
        if not self.config.drop_duplicates:
            return None
        return ("drop_duplicates", DuplicateCleaner())

    def _step_drop_high_missing(self) -> Optional[PipelineStep]:
        if not self.config.drop_high_missing:
            return None
        return ("drop_high_missing", self._make_drop_high_missing_transformer())

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
        creator = RatioFeatureCreator(max_pairs=self.config.ratios_max_pairs)
        return ("ratios", creator)

    def _step_log_transform(self) -> Optional[PipelineStep]:
        if not self.config.apply_log_transform:
            return None
        return ("log_transform", log(columns=self.config.transform_columns))

    def _step_boxcox(self) -> Optional[PipelineStep]:
        if not self.config.apply_boxcox:
            return None
        transformer = boxcox(
            columns=self.config.transform_columns,
            lambda_=self.config.transform_lambda,
        )
        return ("boxcox", transformer)

    def _step_yeojohnson(self) -> Optional[PipelineStep]:
        if not self.config.apply_yeojohnson:
            return None
        transformer = yeojohnson(
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
            reducer = LDAReducer(n_components=self.config.reduction_components)
            return ("lda", reducer)
        return None

    def _make_drop_high_missing_transformer(self) -> FunctionTransformer:
        threshold = self.config.high_missing_threshold
        verbose = self.config.verbose

        def drop_high_missing(X: pd.DataFrame) -> pd.DataFrame:
            missing_pct = X.isnull().mean()
            cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
            if cols_to_drop and verbose:
                print(f"Dropping columns with > {threshold * 100}% missing: {cols_to_drop}")
            return X.drop(columns=cols_to_drop) if cols_to_drop else X

        return FunctionTransformer(drop_high_missing)

    # -- Balancing (hors pipeline sklearn) -----------------------------------

    def apply_balancing(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """À appliquer séparément, avant l'entraînement du modèle."""
        method = self.config.balancing_method
        if method == BalancingMethod.NONE:
            return X, y

        balancer = ClassBalancer(
            method=method,
            sampling_strategy=self.config.balancing_sampling_strategy,
            random_state=self.config.balancing_random_state,
        )
        return balancer.fit_resample(X, y)

    # -- Détection & introspection -------------------------------------------

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
    """Pipeline builder avec des configurations prêtes à l'emploi."""

    def __init__(self, **kwargs):
        super().__init__(PreprocessingConfig(**kwargs))

    @classmethod
    def create_default(cls) -> "SimplePipelineBuilder":
        return cls(
            imputation_method="median",
            scaling_method="standard",
            encoding_method="onehot",
            outlier_method="iqr",
            outlier_threshold=1.5,
            encoding_sparse=False,
        )

    @classmethod
    def create_robust(cls) -> "SimplePipelineBuilder":
        return cls(
            imputation_method="median",
            scaling_method="robust",
            encoding_method="target",
            outlier_method="iqr",
            outlier_threshold=3.0,
            outlier_action="winsorize",
        )

    @classmethod
    def create_high_performance(cls) -> "SimplePipelineBuilder":
        return cls(
            imputation_method="knn",
            scaling_method="standard",
            encoding_method="catboost",
            outlier_method="isolation_forest",
            outlier_action="winsorize",
            create_polynomial=True,
            polynomial_degree=2,
            polynomial_max_features=20,
            apply_boxcox=True,
        )

    @classmethod
    def create_minimal(cls) -> "SimplePipelineBuilder":
        return cls(
            imputation_method="median",
            scaling_method="standard",
            encoding_method="ordinal",
            outlier_method="none",
            drop_duplicates=False,
            drop_high_missing=False,
        )