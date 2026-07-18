# preprocessing/tabular/reducers.py
from typing import Optional, List, Dict, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, RFE, f_classif, f_regression
from sklearn.preprocessing import LabelEncoder

from ...base import BasePreprocessor
from .._column_utils import select_columns
from ..config import TaskType, FeatureSelectionMethod

from ._selection_utils import make_importance_model, encode_categoricals


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