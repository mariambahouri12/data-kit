# tabular/cleaners.py
import warnings
from typing import Optional, Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

from ..base import BasePreprocessor
from ._column_utils import select_columns
from .config import ImputationMethod, OutlierMethod, OutlierAction


class MissingValueCleaner(BasePreprocessor):
    """Impute ou supprime les valeurs manquantes, séparément pour les
    colonnes numériques (stratégie configurable) et catégorielles
    (toujours 'most_frequent')."""

    def __init__(self,
                 strategy: Union[str, ImputationMethod] = ImputationMethod.MEDIAN,
                 fill_value: Optional[Any] = None,
                 columns: Optional[List[str]] = None,
                 knn_neighbors: int = 5,
                 **kwargs):
        """
        Args:
            strategy: stratégie d'imputation pour les colonnes numériques.
            fill_value: valeur utilisée si strategy == CONSTANT.
            columns: colonnes à traiter (None = toutes).
            knn_neighbors: nombre de voisins si strategy == KNN.
        """
        super().__init__(**kwargs)
        self.strategy = ImputationMethod(strategy) if isinstance(strategy, str) else strategy
        self.fill_value = fill_value
        self.columns = columns
        self.knn_neighbors = knn_neighbors

        self.imputer = None
        self.cat_imputer = None
        self.column_types: Dict[str, List[str]] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        cols_to_impute = select_columns(X, self.columns)

        numeric_cols = X[cols_to_impute].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X[cols_to_impute].select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        self.column_types = {"numeric": numeric_cols, "categorical": categorical_cols}

        if numeric_cols:
            self._fit_numeric_imputer(X[numeric_cols])

        if categorical_cols:
            self.cat_imputer = SimpleImputer(strategy="most_frequent", fill_value="unknown")
            self.cat_imputer.fit(X[categorical_cols])

    def _fit_numeric_imputer(self, X_numeric: pd.DataFrame) -> None:
        if self.strategy == ImputationMethod.DROP:
            return  # géré dans _transform via dropna

        if self.strategy == ImputationMethod.KNN:
            self.imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            self.imputer.fit(X_numeric)
            return

        fill_value = self.fill_value
        if self.strategy == ImputationMethod.CONSTANT and fill_value is None:
            fill_value = 0
            warnings.warn(
                "fill_value is None avec strategy=CONSTANT, utilisation de 0 par défaut.",
                RuntimeWarning,
            )

        self.imputer = SimpleImputer(strategy=self.strategy.value, fill_value=fill_value)
        self.imputer.fit(X_numeric)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        numeric_cols = self.column_types.get("numeric", [])
        if numeric_cols:
            if self.strategy == ImputationMethod.DROP:
                X_copy = X_copy.dropna(subset=numeric_cols)
            elif self.imputer is not None:
                X_copy[numeric_cols] = self.imputer.transform(X_copy[numeric_cols])

        categorical_cols = self.column_types.get("categorical", [])
        if categorical_cols and self.cat_imputer is not None:
            X_copy[categorical_cols] = self.cat_imputer.transform(X_copy[categorical_cols])

        return X_copy


class OutlierCleaner(BasePreprocessor):
    """Détecte et traite les outliers (winsorization ou suppression)."""

    def __init__(self,
                 method: Union[str, OutlierMethod] = OutlierMethod.IQR,
                 threshold: float = 1.5,
                 action: Union[str, OutlierAction] = OutlierAction.WINSORIZE,
                 columns: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.method = OutlierMethod(method) if isinstance(method, str) else method
        self.threshold = threshold
        self.action = OutlierAction(action) if isinstance(action, str) else action
        self.columns = columns
        self.bounds: Dict[str, Dict[str, float]] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        cols_to_clean = select_columns(X, self.columns, dtype_include=[np.number])

        for col in cols_to_clean:
            col_data = X[col].dropna()
            if col_data.empty:
                continue
            self.bounds[col] = self._compute_bounds(col_data)

    def _compute_bounds(self, col_data: pd.Series) -> Dict[str, float]:
        if self.method == OutlierMethod.IQR:
            q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            iqr = q3 - q1
            return {"lower": q1 - self.threshold * iqr, "upper": q3 + self.threshold * iqr}

        if self.method == OutlierMethod.ZSCORE:
            mean, std = col_data.mean(), col_data.std()
            return {"lower": mean - self.threshold * std, "upper": mean + self.threshold * std}

        raise ValueError(f"Méthode de détection d'outliers non supportée : {self.method}")

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        for col, bounds in self.bounds.items():
            if self.action == OutlierAction.WINSORIZE:
                X_copy[col] = X_copy[col].clip(lower=bounds["lower"], upper=bounds["upper"])
            elif self.action == OutlierAction.DROP:
                mask = X_copy[col].between(bounds["lower"], bounds["upper"])
                X_copy = X_copy[mask]

        return X_copy


class DuplicateCleaner(BasePreprocessor):
    """Supprime les lignes dupliquées. Stateless : rien à apprendre au fit."""

    def __init__(self, subset: Optional[List[str]] = None, keep: str = "first", **kwargs):
        super().__init__(**kwargs)
        self.subset = subset
        self.keep = keep

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop_duplicates(subset=self.subset, keep=self.keep)