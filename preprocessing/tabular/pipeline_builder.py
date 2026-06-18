# preprocessing/tabular/pipeline_builder.py
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from .config import (
    PreprocessingConfig, 
    EncodingMethod, 
    ScalingMethod,
    ImputationMethod,
    OutlierMethod,
    BalancingMethod,
    FeatureSelectionMethod,
    OutlierAction,
    TaskType
)
from .detectors import (
    MissingValueDetector, OutlierDetector, CorrelationDetector,
    CardinalityDetector, DuplicateDetector
)
from .cleaners import MissingValueCleaner, OutlierCleaner, DuplicateCleaner
from .encoders import CategoricalEncoder, OrdinalEncoderWrapper
from .scalers import FeatureScaler, PowerTransformerWrapper
from .transformers import (
    LogTransformer, SqrtTransformer, BoxCoxTransformer,
    YeoJohnsonTransformer, PercentileTransformer
)
from .reducers import FeatureSelector, PCAReducer, LDAReducer
from .balancers import ClassBalancer
from .feature_engineering import (
    PolynomialFeatureCreator, InteractionFeatureCreator,
    RatioFeatureCreator, AggregationFeatureCreator,
    DateFeatureCreator
)


class PipelineBuilder:
    """
    Constructeur de pipeline de prétraitement flexible.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.steps = []
        self.detectors = []
        self._build_detectors()
    
    def _build_detectors(self):
        """Construire les détecteurs"""
        self.detectors = [
            MissingValueDetector(threshold=0.05),
            OutlierDetector(method='iqr', threshold=1.5),
            CorrelationDetector(threshold=0.8),
            CardinalityDetector(max_categories=50),
            DuplicateDetector()
        ]
    
    def _get_enum_value(self, value, enum_class, default):
        """
        Récupérer la valeur d'un enum ou d'une chaîne.
        
        Args:
            value: Valeur à convertir (str ou enum)
            enum_class: Classe enum
            default: Valeur par défaut
        
        Returns:
            Valeur enum
        """
        if value is None:
            return default
        if isinstance(value, str):
            return enum_class(value)
        if hasattr(value, 'value'):
            return value
        return value
    
    def build_pipeline(self, 
                       X: Optional[pd.DataFrame] = None,
                       y: Optional[pd.Series] = None) -> Pipeline:
        """
        Construire le pipeline complet.
        
        NOTE: Le balancer n'est PAS inclus dans le pipeline sklearn.
        Utilisez la méthode apply_balancing() séparément.
        """
        steps = []
        
        # 1. Drop duplicates
        if self.config.drop_duplicates:
            steps.append(('drop_duplicates', DuplicateCleaner()))
        
        # 2. Drop high missing columns
        if self.config.drop_high_missing:
            steps.append(('drop_high_missing', self._create_drop_high_missing()))
        
        # 3. Imputation
        imputation_method = self.config.imputation_method
        if imputation_method is None or imputation_method == ImputationMethod.DROP or imputation_method == 'drop':
            pass  # Ne pas ajouter l'imputation
        else:
            # Convertir en valeur de chaîne si nécessaire
            if hasattr(imputation_method, 'value'):
                strategy = imputation_method.value
            else:
                strategy = str(imputation_method)
            
            imputer = MissingValueCleaner(
                strategy=strategy,
                fill_value=self.config.imputation_fill_value,
                columns=self.config.imputation_columns
            )
            steps.append(('imputation', imputer))
        
        # 4. Outlier handling
        outlier_method = self.config.outlier_method
        if outlier_method is not None and outlier_method != OutlierMethod.NONE and outlier_method != 'none':
            if hasattr(outlier_method, 'value'):
                method = outlier_method.value
            else:
                method = str(outlier_method)
            
            if hasattr(self.config.outlier_action, 'value'):
                action = self.config.outlier_action.value
            else:
                action = str(self.config.outlier_action)
            
            outlier_cleaner = OutlierCleaner(
                method=method,
                threshold=self.config.outlier_threshold,
                action=action,
                columns=self.config.outlier_columns
            )
            steps.append(('outlier_handling', outlier_cleaner))
        
        # 5. Feature engineering
        if self.config.create_polynomial:
            steps.append(('polynomial', PolynomialFeatureCreator(
                degree=self.config.polynomial_degree,
                max_features=self.config.polynomial_max_features,
                max_output_features=self.config.polynomial_max_output_features
            )))
        
        if self.config.create_interactions:
            steps.append(('interactions', InteractionFeatureCreator()))
        
        if self.config.create_ratios:
            steps.append(('ratios', RatioFeatureCreator(
                max_pairs=self.config.ratios_max_pairs
            )))
        
        # 6. Transformations
        if self.config.apply_log_transform:
            steps.append(('log_transform', LogTransformer(
                columns=self.config.transform_columns
            )))
        
        if self.config.apply_boxcox:
            steps.append(('boxcox', BoxCoxTransformer(
                columns=self.config.transform_columns,
                lambda_=self.config.transform_lambda
            )))
        
        if self.config.apply_yeojohnson:
            steps.append(('yeojohnson', YeoJohnsonTransformer(
                columns=self.config.transform_columns,
                lambda_=self.config.transform_lambda
            )))
        
        # 7. Encoding
        encoding_method = self.config.encoding_method
        if encoding_method is not None and encoding_method != EncodingMethod.NONE and encoding_method != 'none':
            if hasattr(encoding_method, 'value'):
                method = encoding_method.value
            else:
                method = str(encoding_method)
            
            encoder = CategoricalEncoder(
                method=method,
                columns=self.config.encoding_columns,
                max_categories=self.config.encoding_max_categories,
                min_frequency=self.config.encoding_min_frequency,
                handle_unknown=self.config.encoding_handle_unknown,
                sparse=self.config.encoding_sparse,
                target=y
            )
            steps.append(('encoding', encoder))
        
        # 8. Scaling
        scaling_method = self.config.scaling_method
        if scaling_method is not None and scaling_method != ScalingMethod.NONE and scaling_method != 'none':
            if hasattr(scaling_method, 'value'):
                method = scaling_method.value
            else:
                method = str(scaling_method)
            
            scaler = FeatureScaler(
                method=method,
                columns=self.config.scaling_columns,
                with_mean=self.config.scaling_with_mean,
                with_std=self.config.scaling_with_std
            )
            steps.append(('scaling', scaler))
        
        # 9. Feature selection
        feature_selection_method = self.config.feature_selection_method
        if feature_selection_method is not None and feature_selection_method != FeatureSelectionMethod.NONE and feature_selection_method != 'none':
            if hasattr(feature_selection_method, 'value'):
                method = feature_selection_method.value
            else:
                method = str(feature_selection_method)
            
            selector = FeatureSelector(
                method=method,
                threshold=self.config.feature_selection_threshold,
                k=self.config.feature_selection_k,
                columns=self.config.feature_selection_columns,
                task_type=self.config.task_type.value if hasattr(self.config.task_type, 'value') else str(self.config.task_type)
            )
            steps.append(('feature_selection', selector))
        
        # 10. Dimensionality reduction
        if self.config.reduction_method == 'pca':
            reducer = PCAReducer(
                n_components=self.config.reduction_components,
                variance_ratio=self.config.reduction_variance_ratio
            )
            steps.append(('pca', reducer))
        elif self.config.reduction_method == 'lda':
            reducer = LDAReducer(
                n_components=self.config.reduction_components
            )
            steps.append(('lda', reducer))
        
        # Créer le pipeline
        pipeline = Pipeline(steps)
        
        return pipeline
    
    def apply_balancing(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Appliquer le rééquilibrage séparément.
        À utiliser AVANT l'entraînement du modèle.
        
        Args:
            X: Features
            y: Target
        
        Returns:
            (X_resampled, y_resampled)
        """
        balancing_method = self.config.balancing_method
        if balancing_method is None or balancing_method == BalancingMethod.NONE or balancing_method == 'none':
            return X, y
        
        if hasattr(balancing_method, 'value'):
            method = balancing_method.value
        else:
            method = str(balancing_method)
        
        balancer = ClassBalancer(
            method=method,
            sampling_strategy=self.config.balancing_sampling_strategy,
            random_state=self.config.balancing_random_state
        )
        
        return balancer.fit_resample(X, y)
    
    def _create_drop_high_missing(self):
        """Créer un transformateur pour supprimer les colonnes avec trop de valeurs manquantes"""
        threshold = self.config.high_missing_threshold
        
        def drop_high_missing(X: pd.DataFrame) -> pd.DataFrame:
            missing_pct = X.isnull().mean()
            cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
            
            if cols_to_drop and self.config.verbose:
                print(f"Dropping columns with > {threshold*100}% missing: {cols_to_drop}")
            
            return X.drop(columns=cols_to_drop) if cols_to_drop else X
        
        return FunctionTransformer(drop_high_missing)
    
    def build_detection_pipeline(self) -> Pipeline:
        """Construire un pipeline de détection"""
        return make_pipeline(*self.detectors)
    
    def get_step_names(self) -> List[str]:
        """Obtenir les noms des étapes du pipeline"""
        return self.config.get_active_steps()
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Obtenir un résumé du pipeline"""
        return {
            'steps': self.get_step_names(),
            'n_steps': len(self.get_step_names()),
            'config': self.config.to_dict()
        }


class SimplePipelineBuilder(PipelineBuilder):
    """Constructeur de pipeline simplifié"""
    
    def __init__(self, **kwargs):
        config = PreprocessingConfig(**kwargs)
        super().__init__(config)
    
    @classmethod
    def create_default(cls) -> 'SimplePipelineBuilder':
        return cls(
            imputation_method='median',
            scaling_method='standard',
            encoding_method='onehot',
            outlier_method='iqr',
            outlier_threshold=1.5,
            encoding_sparse=False
        )
    
    @classmethod
    def create_robust(cls) -> 'SimplePipelineBuilder':
        return cls(
            imputation_method='median',
            scaling_method='robust',
            encoding_method='target',
            outlier_method='iqr',
            outlier_threshold=3.0,
            outlier_action='winsorize'
        )
    
    @classmethod
    def create_high_performance(cls) -> 'SimplePipelineBuilder':
        return cls(
            imputation_method='knn',
            scaling_method='standard',
            encoding_method='catboost',
            outlier_method='isolation_forest',
            outlier_action='winsorize',
            create_polynomial=True,
            polynomial_degree=2,
            polynomial_max_features=20,
            apply_boxcox=True
        )
    
    @classmethod
    def create_minimal(cls) -> 'SimplePipelineBuilder':
        return cls(
            imputation_method='median',
            scaling_method='standard',
            encoding_method='ordinal',
            outlier_method='none',
            drop_duplicates=False,
            drop_high_missing=False
        )