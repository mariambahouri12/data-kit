# preprocessing/tabular/pipeline_builder.py
# Remarque : une limite avec la configuration actuelle : on applique la meme strategie pour toutes les colonnes -> si on veux , ca va etre plus complexe mais plus flexible -> à faire comme amélioration apres
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
    Flexible preprocessing pipeline builder
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):  # fonction constructeur s'éxecute automatiquement quand on crée un nouvel objet
        # le mot self répresente l'objet qu'on est en train de creer
        # Optional signfie ce parametre peut aussi etre None , None si l'utilisateur ne donne rien donc valeur par défaut 
        self.config = config or PreprocessingConfig() # si l'utilisateur a donné une config on l'utilise sinon on en crée une nouvelle par défaut 
        self.steps = [] # pour stocker les etapes apres
        self.detectors = [] # pour ajouter les détecteurs 
        self._build_detectors() # le _ indique que c'est une méthode privée(interne)
    
    def _build_detectors(self): # creer les détecteurs et les stocke dans self.detectors
        """Build the detectors"""
        self.detectors = [
            MissingValueDetector(threshold=0.05), # detecter valeurs manquantes >5%
            OutlierDetector(method='iqr', threshold=1.5),
            CorrelationDetector(threshold=0.8), # detécte corrélation >0.8
            CardinalityDetector(max_categories=50), # détecte trop de catégories
            DuplicateDetector()  # detecte doublons 
        ]
    
    def _get_enum_value(self, value, enum_class, default): # cette fonction prend une valeur qui peut etre Enum , string ou None et la convertit en Enum valide
       # value : valeur à convertir , enum_class la classe Enum à utiliser, default str ou Enum : valeur par défaut si value est None
       # on utilise cette fonction pour uniformiser -> pour toujours avoir Enum
       # cette fonction est privée car utilisée uniquement à l'intérieur de la classe PipelineBuilder
        """
       Get the value from an enum or a string.

           Args:
             value: Value to convert (str or enum)
             enum_class: Enum class
            default: Default value

         Returns:
            Enum value
        """
        #
        if value is None:
            return default
        if isinstance(value, str):
            return enum_class(value)
        if hasattr(value, 'value'): # verifier si value est un Enum ( ou a un attribut 'value')
            # hasattr verifie si l'objet possede un attribut avec ce nom -> si oui c'est un Enum - > on le retourne  tel quel
            return value
        return value # c'est le cas par défaut 
    # cette fonction maintenant construit un pipeline sklearn
    def build_pipeline(self, 
                       X: Optional[pd.DataFrame] = None, # les donnees d'entree
                       y: Optional[pd.Series] = None) -> Pipeline: # ce qu'on veut prédire
        """
        Build the complete pipeline.
        
        NOTE: The balancer is NOT included in the sklearn pipeline
        Use the apply_balancing() method separately
        """
        steps = [] # pour stocker les étapes
        
        # 1. Drop duplicates
        if self.config.drop_duplicates:
            steps.append(('drop_duplicates', DuplicateCleaner()))
        
        # 2. Drop high missing columns
        if self.config.drop_high_missing:
            steps.append(('drop_high_missing', self._create_drop_high_missing()))
        
        # 3. Imputation
        imputation_method = self.config.imputation_method
        if imputation_method is None or imputation_method == ImputationMethod.DROP or imputation_method == 'drop': # qui ecrit ca exactement , plus on a dit qu'on va utiliser toujours Enum donc pourquoi drop?
            pass  # Ne pas ajouter l'imputation
        else:
            # Convertir en valeur de chaîne si nécessaire -> ici on converit en string car la classe MissingValueCleaner attendent ( et les bibliothèques sklearn) attendent une string en paramètre
            if hasattr(imputation_method, 'value'):
                strategy = imputation_method.value
            else:
                strategy = str(imputation_method)
            
            imputer = MissingValueCleaner(
                strategy=strategy,
                fill_value=self.config.imputation_fill_value,  # une valeur qu'on va utiliser pour remplacer les valeurs manquantes quand on utilise la stratégie constant
                columns=self.config.imputation_columns # les colonnes à traiter 
            )
            steps.append(('imputation', imputer))
        
        # 4. Outlier handling
        outlier_method = self.config.outlier_method # pour detecter
        if outlier_method is not None and outlier_method != OutlierMethod.NONE and outlier_method != 'none':
            if hasattr(outlier_method, 'value'):
                method = outlier_method.value
            else:
                method = str(outlier_method)
            
            if hasattr(self.config.outlier_action, 'value'): # pour traiter
                action = self.config.outlier_action.value
            else:
                action = str(self.config.outlier_action)
            
            outlier_cleaner = OutlierCleaner(
                method=method,
                threshold=self.config.outlier_threshold, # le seuil de détection
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
        Apply the balancing separately.
        To be used BEFORE model training.
        
        Args:
            X: Features
            y: Target
        
        Returns:
            (X_resampled, y_resampled)
        """
        balancing_method = self.config.balancing_method
        if balancing_method is None or balancing_method == BalancingMethod.NONE or balancing_method == 'none':
            return X, y # retourner les données telles quelles 
        
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
        """Create a transformer to remove columns with too many missing values."""
        threshold = self.config.high_missing_threshold
        
        def drop_high_missing(X: pd.DataFrame) -> pd.DataFrame:
            missing_pct = X.isnull().mean()
            cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
            
            if cols_to_drop and self.config.verbose:
                print(f"Dropping columns with > {threshold*100}% missing: {cols_to_drop}")
            
            return X.drop(columns=cols_to_drop) if cols_to_drop else X
        
        return FunctionTransformer(drop_high_missing)
    
    def build_detection_pipeline(self) -> Pipeline:
        """Build a detection pipeline"""
        return make_pipeline(*self.detectors) # c'est une fonction de sklearn qui crée un pipeline plus facilement que la classe Pipeline()
    
    def get_step_names(self) -> List[str]:
        """Get the names of the pipeline steps"""
        return self.config.get_active_steps()
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline"""
        return {
            'steps': self.get_step_names(),
            'n_steps': len(self.get_step_names()),
            'config': self.config.to_dict()
        }


class SimplePipelineBuilder(PipelineBuilder): # version de PipelineBuiled : configurations pretes à l'emploi
    """Simplified pipeline builder"""
    
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
    
# config.py -> PipelineBuilder -> Transformers / Cleaners / Encoders -> Pipeline sklearn -> fit() / trasnform
# config.py est le plan du pipeline et pipelineBuilder construit réellement le pipeline à partir de ce plan
# self : methode d'instance : l'objet (une instance de la classe), cls : méthode de classe : la classe (pas une instance)