# preprocessing/tabular/config.py

from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field, fields, asdict
from enum import Enum
import json


class ImputationMethod(str, Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    MODE = 'most_frequent'
    CONSTANT = 'constant'
    KNN = 'knn'
    MICE = 'mice'
    DROP = 'drop'


class ScalingMethod(str, Enum):
    STANDARD = 'standard'
    MINMAX = 'minmax'
    ROBUST = 'robust'
    MAXABS = 'maxabs'
    QUANTILE = 'quantile'
    POWER = 'power'
    NONE = 'none'


class EncodingMethod(str, Enum):
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
    IQR = 'iqr'
    ZSCORE = 'zscore'
    ISOLATION_FOREST = 'isolation_forest'
    DBSCAN = 'dbscan'
    NONE = 'none'


class OutlierAction(str, Enum):
    WINSORIZE = 'winsorize'
    DROP = 'drop'


class BalancingMethod(str, Enum):
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
    VARIANCE = 'variance'
    CORRELATION = 'correlation'
    IMPORTANCE = 'importance'
    RFE = 'rfe'
    NONE = 'none'


class TaskType(str, Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


# Un seul endroit qui sait "ce champ est un enum, et lequel".
# to_dict / from_dict / validation en dérivent automatiquement,
# donc un champ oublié devient impossible plutôt qu'un bug silencieux.
_ENUM_FIELDS: Dict[str, Type[Enum]] = {
    'task_type': TaskType,
    'imputation_method': ImputationMethod,
    'scaling_method': ScalingMethod,
    'encoding_method': EncodingMethod,
    'outlier_method': OutlierMethod,
    'outlier_action': OutlierAction,
    'balancing_method': BalancingMethod,
    'feature_selection_method': FeatureSelectionMethod,
}


@dataclass
class PreprocessingConfig:
    """Configuration complète du pipeline de prétraitement."""

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
    # FIX (dead code éliminé) : `balancing_target` a été retiré. Ce champ
    # n'était jamais lu ni par ClassBalancer, ni par pipeline_builder, ni
    # par orchestrator — la colonne cible est toujours résolue via
    # detect_target_column(). Le garder aurait laissé croire à un
    # utilisateur de l'API qu'il pouvait forcer la target du balancing,
    # alors que ce n'était le cas nulle part dans le code.
    balancing_method: BalancingMethod = BalancingMethod.NONE
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

    # -- Normalisation ----------------------------------------------------

    def __post_init__(self) -> None:
        """Normalise les champs enum : accepte une string ou un Enum en entrée,
        garantit un Enum en sortie. Élimine le besoin, partout ailleurs dans le
        code, de tester hasattr(x, 'value') avant de lire une méthode de config.
        """
        for field_name, enum_cls in _ENUM_FIELDS.items():
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, enum_cls(value))

    # -- Sérialisation --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire (enums -> str)."""
        raw = asdict(self)
        for field_name in _ENUM_FIELDS:
            if raw.get(field_name) is not None:
                value = raw[field_name]
                raw[field_name] = value.value if isinstance(value, Enum) else value
        return raw

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreprocessingConfig':
        """Crée une configuration depuis un dictionnaire (str -> enums)."""
        valid_keys = {f.name for f in fields(cls)}
        clean_data = {k: v for k, v in data.items() if k in valid_keys}

        for field_name, enum_cls in _ENUM_FIELDS.items():
            value = clean_data.get(field_name)
            if isinstance(value, str):
                clean_data[field_name] = enum_cls(value)

        return cls(**clean_data)

    def save(self, filepath: str) -> None:
        """Sauvegarde la configuration dans un fichier JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'PreprocessingConfig':
        """Charge une configuration depuis un fichier JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    # -- Introspection ----------------------------------------------------

    def get_active_steps(self) -> List[str]:
        """Liste les étapes du pipeline effectivement actives."""
        steps = []

        if self.imputation_method != ImputationMethod.DROP:
            steps.append('imputation')
        if self.scaling_method != ScalingMethod.NONE:
            steps.append('scaling')
        if self.encoding_method != EncodingMethod.NONE:
            steps.append('encoding')
        if self.outlier_method != OutlierMethod.NONE:
            steps.append('outlier_handling')
        if self.balancing_method != BalancingMethod.NONE and not self.balancing_apply_before_pipeline:
            steps.append('balancing')
        if self.feature_selection_method != FeatureSelectionMethod.NONE:
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