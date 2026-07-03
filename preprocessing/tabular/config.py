# preprocessing/tabular/config.py

# preprocessing/tabular/config.py
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json


class ImputationMethod(str, Enum):
    """Méthodes d'imputation des valeurs manquantes"""
    MEAN = 'mean'
    MEDIAN = 'median'
    MODE = 'most_frequent'
    CONSTANT = 'constant'
    KNN = 'knn'
    MICE = 'mice'
    DROP = 'drop'


class ScalingMethod(str, Enum):
    """Méthodes de normalisation/standardisation"""
    STANDARD = 'standard'
    MINMAX = 'minmax'
    ROBUST = 'robust'
    MAXABS = 'maxabs'
    QUANTILE = 'quantile'
    POWER = 'power'
    NONE = 'none'


class EncodingMethod(str, Enum):
    """Méthodes d'encodage des variables catégorielles"""
    ONE_HOT = 'onehot'
    LABEL = 'label'
    TARGET = 'target'
    FREQUENCY = 'frequency'
    BINARY = 'binary'
    CATBOOST = 'catboost'
    HASH = 'hash'
    ORDINAL = 'ordinal'
    NONE = 'none'


class OutlierMethod(str, Enum):
    """Méthodes de détection des outliers"""
    IQR = 'iqr'
    ZSCORE = 'zscore'
    ISOLATION_FOREST = 'isolation_forest'
    DBSCAN = 'dbscan'
    NONE = 'none'


class OutlierAction(str, Enum):
    """Actions à prendre pour les outliers"""
    WINSORIZE = 'winsorize'
    DROP = 'drop'


class BalancingMethod(str, Enum):
    """Méthodes de rééquilibrage des classes"""
    SMOTE = 'smote'
    ADASYN = 'adasyn'
    RANDOM_OVER = 'random_over'
    RANDOM_UNDER = 'random_under'
    TOMEK = 'tomek'
    ENN = 'enn'
    SMOTE_TOMEK = 'smote_tomek'
    SMOTE_ENN = 'smote_enn'
    NONE = 'none'


class FeatureSelectionMethod(str, Enum):
    """Méthodes de sélection de features"""
    VARIANCE = 'variance'
    CORRELATION = 'correlation'
    IMPORTANCE = 'importance'
    RFE = 'rfe'
    NONE = 'none'


class TaskType(str, Enum):
    """Type de tâche ML"""
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


@dataclass
class PreprocessingConfig:
    """
    Configuration complète du pipeline de prétraitement.
    """
    
    # === Task Type ===
    task_type: TaskType = TaskType.CLASSIFICATION
    
    # === Imputation ===
    imputation_method: ImputationMethod = ImputationMethod.MEDIAN
    imputation_columns: Optional[List[str]] = None
    imputation_fill_value: Optional[Any] = None
    imputation_knn_neighbors: int = 5
    
    # === Scaling ===
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    scaling_columns: Optional[List[str]] = None
    scaling_with_mean: bool = True
    scaling_with_std: bool = True
    
    # === Encoding ===
    encoding_method: EncodingMethod = EncodingMethod.ONE_HOT
    encoding_columns: Optional[List[str]] = None
    encoding_max_categories: int = 50
    encoding_min_frequency: float = 0.01
    encoding_handle_unknown: str = 'ignore'
    encoding_sparse: bool = True
    
    # === Outliers ===
    outlier_method: OutlierMethod = OutlierMethod.IQR
    outlier_threshold: float = 1.5
    outlier_action: OutlierAction = OutlierAction.WINSORIZE
    outlier_columns: Optional[List[str]] = None
    
    # === Balancing ===
    balancing_method: BalancingMethod = BalancingMethod.NONE
    balancing_target: Optional[str] = None
    balancing_sampling_strategy: Optional[Dict] = None
    balancing_random_state: int = 42
    balancing_apply_before_pipeline: bool = True
    
    # === Feature Selection ===
    feature_selection_method: FeatureSelectionMethod = FeatureSelectionMethod.NONE
    feature_selection_threshold: float = 0.01
    feature_selection_k: Optional[int] = None
    feature_selection_columns: Optional[List[str]] = None
    
    # === Transformations ===
    apply_log_transform: bool = False
    apply_boxcox: bool = False
    apply_yeojohnson: bool = False
    transform_columns: Optional[List[str]] = None
    transform_lambda: Optional[float] = None
    
    # === Feature Engineering ===
    create_polynomial: bool = False
    polynomial_degree: int = 2
    polynomial_max_features: int = 50
    polynomial_max_output_features: int = 5000
    create_interactions: bool = False
    create_ratios: bool = False
    ratios_max_pairs: int = 100
    create_aggregations: bool = False
    engineering_columns: Optional[List[str]] = None
    
    # === Dimensionality Reduction ===
    reduction_method: Optional[str] = None
    reduction_components: Optional[int] = None
    reduction_variance_ratio: float = 0.95
    
    # === General ===
    drop_duplicates: bool = True
    drop_high_missing: bool = True
    high_missing_threshold: float = 0.8
    random_state: int = 42
    verbose: bool = True
    
    # ✅ Fonction utilitaire pour extraire la valeur d'un Enum ou d'une string
    def _get_enum_value(self, value):
        """Convertit un Enum ou une string en valeur string"""
        if value is None:
            return None
        if hasattr(value, 'value'):
            return value.value
        return str(value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir la configuration en dictionnaire"""
        return {
            'task_type': self._get_enum_value(self.task_type),
            'imputation_method': self._get_enum_value(self.imputation_method),
            'imputation_columns': self.imputation_columns,
            'imputation_fill_value': self.imputation_fill_value,
            'imputation_knn_neighbors': self.imputation_knn_neighbors,
            'scaling_method': self._get_enum_value(self.scaling_method),
            'scaling_columns': self.scaling_columns,
            'scaling_with_mean': self.scaling_with_mean,
            'scaling_with_std': self.scaling_with_std,
            'encoding_method': self._get_enum_value(self.encoding_method),
            'encoding_columns': self.encoding_columns,
            'encoding_max_categories': self.encoding_max_categories,
            'encoding_min_frequency': self.encoding_min_frequency,
            'encoding_handle_unknown': self.encoding_handle_unknown,
            'encoding_sparse': self.encoding_sparse,
            'outlier_method': self._get_enum_value(self.outlier_method),
            'outlier_threshold': self.outlier_threshold,
            'outlier_action': self._get_enum_value(self.outlier_action),
            'outlier_columns': self.outlier_columns,
            'balancing_method': self._get_enum_value(self.balancing_method),
            'balancing_target': self.balancing_target,
            'balancing_sampling_strategy': self.balancing_sampling_strategy,
            'balancing_random_state': self.balancing_random_state,
            'balancing_apply_before_pipeline': self.balancing_apply_before_pipeline,
            'feature_selection_method': self._get_enum_value(self.feature_selection_method),
            'feature_selection_threshold': self.feature_selection_threshold,
            'feature_selection_k': self.feature_selection_k,
            'feature_selection_columns': self.feature_selection_columns,
            'apply_log_transform': self.apply_log_transform,
            'apply_boxcox': self.apply_boxcox,
            'apply_yeojohnson': self.apply_yeojohnson,
            'transform_columns': self.transform_columns,
            'transform_lambda': self.transform_lambda,
            'create_polynomial': self.create_polynomial,
            'polynomial_degree': self.polynomial_degree,
            'polynomial_max_features': self.polynomial_max_features,
            'polynomial_max_output_features': self.polynomial_max_output_features,
            'create_interactions': self.create_interactions,
            'create_ratios': self.create_ratios,
            'ratios_max_pairs': self.ratios_max_pairs,
            'create_aggregations': self.create_aggregations,
            'engineering_columns': self.engineering_columns,
            'reduction_method': self.reduction_method,
            'reduction_components': self.reduction_components,
            'reduction_variance_ratio': self.reduction_variance_ratio,
            'drop_duplicates': self.drop_duplicates,
            'drop_high_missing': self.drop_high_missing,
            'high_missing_threshold': self.high_missing_threshold,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreprocessingConfig':
        """Créer une configuration depuis un dictionnaire"""
        if 'task_type' in data and isinstance(data['task_type'], str):
            data['task_type'] = TaskType(data['task_type'])
        if 'imputation_method' in data and isinstance(data['imputation_method'], str):
            data['imputation_method'] = ImputationMethod(data['imputation_method'])
        if 'scaling_method' in data and isinstance(data['scaling_method'], str):
            data['scaling_method'] = ScalingMethod(data['scaling_method'])
        if 'encoding_method' in data and isinstance(data['encoding_method'], str):
            data['encoding_method'] = EncodingMethod(data['encoding_method'])
        if 'outlier_method' in data and isinstance(data['outlier_method'], str):
            data['outlier_method'] = OutlierMethod(data['outlier_method'])
        if 'outlier_action' in data and isinstance(data['outlier_action'], str):
            data['outlier_action'] = OutlierAction(data['outlier_action'])
        if 'balancing_method' in data and isinstance(data['balancing_method'], str):
            data['balancing_method'] = BalancingMethod(data['balancing_method'])
        if 'feature_selection_method' in data and isinstance(data['feature_selection_method'], str):
            data['feature_selection_method'] = FeatureSelectionMethod(data['feature_selection_method'])
        
        return cls(**data)
    
    def save(self, filepath: str):
        """Sauvegarder la configuration dans un fichier JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'PreprocessingConfig':
        """Charger une configuration depuis un fichier JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    # ✅ CORRIGÉ : get_active_steps() gère les strings et les Enums
    def get_active_steps(self) -> List[str]:
        """Obtenir la liste des étapes actives"""
        steps = []
        
        # Extraire les valeurs (gère les Enums et les strings)
        imputation = self._get_enum_value(self.imputation_method)
        scaling = self._get_enum_value(self.scaling_method)
        encoding = self._get_enum_value(self.encoding_method)
        outlier = self._get_enum_value(self.outlier_method)
        balancing = self._get_enum_value(self.balancing_method)
        feature_selection = self._get_enum_value(self.feature_selection_method)
        
        if imputation != 'drop':
            steps.append('imputation')
        if scaling != 'none':
            steps.append('scaling')
        if encoding != 'none':
            steps.append('encoding')
        if outlier != 'none':
            steps.append('outlier_handling')
        if balancing != 'none' and not self.balancing_apply_before_pipeline:
            steps.append('balancing')
        if feature_selection != 'none':
            steps.append('feature_selection')
        if self.apply_log_transform or self.apply_boxcox or self.apply_yeojohnson:
            steps.append('transformation')
        if self.create_polynomial or self.create_interactions or self.create_ratios:
            steps.append('feature_engineering')
        if self.reduction_method:
            steps.append('dimensionality_reduction')
        if self.drop_duplicates:
            steps.append('drop_duplicates')
        if self.drop_high_missing:
            steps.append('drop_high_missing')
        
        return steps