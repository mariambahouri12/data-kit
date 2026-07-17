# preprocessing/tabular/reducers.py
from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold, RFE, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from ..base import BasePreprocessor
from ._column_utils import select_columns
from .config import TaskType, FeatureSelectionMethod


def make_importance_model(task_type: TaskType):
    """Modèle utilisé pour les méthodes basées sur l'importance des features (importance, RFE)."""
    if task_type == TaskType.CLASSIFICATION:
        return RandomForestClassifier(n_estimators=100, random_state=42)
    return RandomForestRegressor(n_estimators=100, random_state=42)


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Encode les colonnes catégorielles en codes entiers (pour modèles sklearn qui l'exigent)."""
    X_encoded = X.copy()
    cat_cols = X_encoded.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X_encoded[col] = X_encoded[col].astype("category").cat.codes
    return X_encoded


class FeatureSelector(BasePreprocessor):
    """
    Sélection de features par différentes méthodes :
    - variance : élimine les colonnes numériques à variance quasi nulle.
    - correlation : garde les colonnes les plus corrélées à la target
      (F-test ANOVA en classification, Pearson en régression).
    - importance : garde les colonnes les plus importantes selon une RandomForest.
    - rfe : élimination récursive de features via une RandomForest.
    """

    _FIT_METHODS = {
        FeatureSelectionMethod.VARIANCE: "_fit_variance",
        FeatureSelectionMethod.CORRELATION: "_fit_correlation",
        FeatureSelectionMethod.IMPORTANCE: "_fit_importance",
        FeatureSelectionMethod.RFE: "_fit_rfe",
    }

    def __init__(self,
                 method: Union[str, FeatureSelectionMethod] = FeatureSelectionMethod.VARIANCE,
                 threshold: float = 0.01,
                 k: Optional[int] = None,
                 columns: Optional[List[str]] = None,
                 task_type: Union[str, TaskType] = TaskType.CLASSIFICATION,
                 **kwargs):
        super().__init__(**kwargs)
        self.method = FeatureSelectionMethod(method) if isinstance(method, str) else method
        self.threshold = threshold
        self.k = k
        self.columns = columns
        self.task_type = TaskType(task_type) if isinstance(task_type, str) else task_type

        self.selector = None
        self.selected_features: List[str] = []
        self.feature_importances: Dict[str, float] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        cols_to_select = select_columns(X, self.columns)
        X_selected = X[cols_to_select]

        if self.method == FeatureSelectionMethod.NONE:
            self.selected_features = cols_to_select
            return

        fit_method_name = self._FIT_METHODS[self.method]
        getattr(self, fit_method_name)(X_selected, y)

    def _fit_variance(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if numeric_cols.empty:
            self.selected_features = X.columns.tolist()
            return

        self.selector = VarianceThreshold(threshold=self.threshold)
        self.selector.fit(X[numeric_cols])

        mask = self.selector.get_support()
        selected_numeric = numeric_cols[mask].tolist()
        non_numeric = [c for c in X.columns if c not in numeric_cols]
        self.selected_features = selected_numeric + non_numeric

    def _fit_correlation(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Sélectionne les colonnes les plus corrélées à la target :
        F-test ANOVA (f_classif) en classification, corrélation de Pearson (f_regression) en régression."""
        if y is None:
            self.selected_features = X.columns.tolist()
            return

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            self.selected_features = X.columns.tolist()
            return

        y_encoded = self._encode_target_if_needed(y)
        score_fn = f_classif if self.task_type == TaskType.CLASSIFICATION else f_regression
        f_scores, _ = score_fn(X[numeric_cols], y_encoded)

        sorted_cols = sorted(zip(numeric_cols, f_scores), key=lambda x: x[1], reverse=True)
        selected = self._top_k_or_threshold(sorted_cols)

        non_numeric = [c for c in X.columns if c not in numeric_cols]
        self.selected_features = selected + non_numeric
        self.feature_importances = dict(sorted_cols)

    def _fit_importance(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        if y is None:
            self.selected_features = X.columns.tolist()
            return

        model = make_importance_model(self.task_type)
        X_encoded = encode_categoricals(X)
        model.fit(X_encoded, y)

        sorted_features = sorted(
            zip(X_encoded.columns, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        self.selected_features = self._top_k_or_threshold(sorted_features)
        self.feature_importances = dict(sorted_features)

    def _fit_rfe(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        if y is None:
            self.selected_features = X.columns.tolist()
            return

        model = make_importance_model(self.task_type)
        X_encoded = encode_categoricals(X)  # nécessaire : RFE fit un modèle sklearn en interne
        n_features = self.k if self.k else max(1, len(X_encoded.columns) // 2)

        self.selector = RFE(model, n_features_to_select=n_features)
        self.selector.fit(X_encoded, y)

        mask = self.selector.get_support()
        self.selected_features = X_encoded.columns[mask].tolist()

    def _top_k_or_threshold(self, sorted_scores: List[tuple]) -> List[str]:
        if self.k:
            return [col for col, _ in sorted_scores[:self.k]]
        return [col for col, score in sorted_scores if score >= self.threshold]

    @staticmethod
    def _encode_target_if_needed(y: pd.Series):
        if y.dtype == "object" or isinstance(y.dtype, pd.CategoricalDtype):
            return LabelEncoder().fit_transform(y)
        return y

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features:
            return X.copy()
        return X[self.selected_features]


class PCAReducer(BasePreprocessor):
    """Réduction de dimension par PCA sur les colonnes numériques."""

    def __init__(self,
                 n_components: Optional[int] = None,
                 variance_ratio: float = 0.95,
                 columns: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        self.columns = columns

        self.pca = None
        self.feature_names: List[str] = []
        self.columns_to_reduce: List[str] = []

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.columns_to_reduce = select_columns(X, self.columns, dtype_include=[np.number])
        if not self.columns_to_reduce:
            return

        n_components = self.n_components if self.n_components is not None else self.variance_ratio
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X[self.columns_to_reduce])

        self.feature_names = [f"PC{i + 1}" for i in range(self.pca.n_components_)]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns_to_reduce:
            return X.copy()

        pca_result = self.pca.transform(X[self.columns_to_reduce])
        pca_df = pd.DataFrame(pca_result, columns=self.feature_names, index=X.index)

        return pd.concat([X.drop(columns=self.columns_to_reduce), pca_df], axis=1)

    def get_explained_variance(self) -> Dict[str, Any]:
        if self.pca is None:
            return {}
        ratios = self.pca.explained_variance_ratio_
        return {
            "explained_variance_ratio": ratios.tolist(),
            "cumulative_variance": ratios.cumsum().tolist(),
            "total_variance": float(ratios.sum()),
        }


class LDAReducer(BasePreprocessor):
    """Réduction de dimension supervisée par LDA (nécessite une target catégorielle)."""

    def __init__(self,
                 n_components: Optional[int] = None,
                 columns: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.columns = columns

        self.lda = None
        self.feature_names: List[str] = []
        self.columns_to_reduce: List[str] = []

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        if y is None:
            raise ValueError("LDA requires target variable")

        self.columns_to_reduce = select_columns(X, self.columns, dtype_include=[np.number])
        if not self.columns_to_reduce:
            return

        n_classes = y.nunique()
        max_components = n_classes - 1
        requested = self.n_components if self.n_components is not None else max_components
        n_components = min(requested, max_components)

        if n_components < 1:
            raise ValueError(f"Not enough classes for LDA. Need at least 2 classes, got {n_classes}")

        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        self.lda.fit(X[self.columns_to_reduce], y)

        self.feature_names = [f"LD{i + 1}" for i in range(self.lda.n_components)]

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns_to_reduce:
            return X.copy()

        lda_result = self.lda.transform(X[self.columns_to_reduce])
        lda_df = pd.DataFrame(lda_result, columns=self.feature_names, index=X.index)

        return pd.concat([X.drop(columns=self.columns_to_reduce), lda_df], axis=1)