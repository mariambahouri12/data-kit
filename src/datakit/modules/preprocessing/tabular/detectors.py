# preprocessing/tabular/detectors.py
import warnings
from typing import Dict, Any, Optional, List, Union

import numpy as np
import pandas as pd

from ..base import BaseDetector
from ._column_utils import select_columns
from .config import OutlierMethod


class MissingValueDetector(BaseDetector):
    """Détecte les colonnes avec un taux de valeurs manquantes préoccupant."""

    HIGH_SEVERITY_THRESHOLD_PCT = 20.0

    def __init__(self, threshold: float = 0.05, **kwargs):
        """
        Args:
            threshold: Tolérance, en fraction (défaut : 0.05 = 5%).
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.missing_stats: Dict[str, Any] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.problems = []
        self.missing_stats = {
            "total_missing": int(X.isnull().sum().sum()),
            "total_cells": X.size,
            "missing_percentage": (X.isnull().sum().sum() / X.size) * 100,
            "columns": {},
        }

        for col in X.columns:
            missing_count = X[col].isnull().sum()
            missing_pct = (missing_count / len(X)) * 100
            self.missing_stats["columns"][col] = {
                "missing_count": int(missing_count),
                "missing_percentage": missing_pct,
            }

            if missing_pct > self.threshold * 100:
                severity = "high" if missing_pct > self.HIGH_SEVERITY_THRESHOLD_PCT else "medium"
                self.problems.append({
                    "column": col,
                    "description": f"{missing_pct:.1f}% de valeurs manquantes",
                    "severity": severity,
                    "suggestion": self._suggest_imputation(missing_pct),
                })

    @staticmethod
    def _suggest_imputation(missing_pct: float) -> str:
        if missing_pct < 5:
            return "Supprimer les lignes ou imputer par la moyenne"
        if missing_pct < 20:
            return "Imputer par la médiane (robuste) ou KNN"
        return "Supprimer la colonne ou imputation avancée (MICE)"

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class OutlierDetector(BaseDetector):
    """Détecte la proportion de valeurs aberrantes par colonne numérique."""

    HIGH_SEVERITY_THRESHOLD_PCT = 10.0

    def __init__(self, method: Union[str, OutlierMethod] = OutlierMethod.IQR,
                 threshold: float = 1.5, **kwargs):
        """
        Args:
            method: IQR ou ZSCORE.
            threshold: Seuil (1.5 typique pour IQR, 3 pour z-score).
        """
        super().__init__(**kwargs)
        self.method = OutlierMethod(method) if isinstance(method, str) else method
        self.threshold = threshold
        self.outlier_stats: Dict[str, Dict[str, float]] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.problems = []
        self.outlier_stats = {}
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_data = X[col].dropna()
            if col_data.empty:
                continue

            n_outliers = self._count_outliers(X[col], col_data)
            if n_outliers is None:
                continue

            outlier_pct = (n_outliers / len(X)) * 100
            self.outlier_stats[col] = {"n_outliers": n_outliers, "percentage": outlier_pct}

            if n_outliers > 0:
                severity = "high" if outlier_pct > self.HIGH_SEVERITY_THRESHOLD_PCT else "medium"
                self.problems.append({
                    "column": col,
                    "description": f"{n_outliers} outliers ({outlier_pct:.1f}%)",
                    "severity": severity,
                    "suggestion": self._suggest_treatment(),
                })

    def _count_outliers(self, full_column: pd.Series, non_null_data: pd.Series) -> Optional[int]:
        """Retourne le nombre d'outliers, ou None si le calcul n'est pas possible (ex: std=0)."""
        if self.method == OutlierMethod.IQR:
            q1, q3 = non_null_data.quantile(0.25), non_null_data.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - self.threshold * iqr, q3 + self.threshold * iqr
            return int(((full_column < lower) | (full_column > upper)).sum())

        if self.method == OutlierMethod.ZSCORE:
            mean, std = non_null_data.mean(), non_null_data.std()
            if std == 0:
                return None
            z_scores = np.abs((full_column - mean) / std)
            return int((z_scores > self.threshold).sum())

        raise ValueError(f"Méthode de détection d'outliers non supportée : {self.method}")

    def _suggest_treatment(self) -> str:
        if self.method == OutlierMethod.IQR:
            return "Winsoriser ou capper les outliers"
        return "Z-score: supprimer ou transformer"

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class CorrelationDetector(BaseDetector):
    """Détecte les paires de colonnes numériques fortement corrélées."""

    def __init__(self, threshold: float = 0.8, **kwargs):
        """
        Args:
            threshold: Seuil de corrélation absolue (défaut : 0.8).
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.correlations: Dict[str, Any] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.problems = []
        self.correlations = {}

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return

        data = X[numeric_cols].dropna()
        if len(data) < 2:
            warnings.warn("Not enough data after dropping NaNs for correlation", RuntimeWarning)
            return

        corr_matrix = data.corr().abs()
        high_corr_pairs = self._find_high_corr_pairs(corr_matrix, numeric_cols)

        self.correlations = {"matrix": corr_matrix, "high_corr_pairs": high_corr_pairs}

        for pair in high_corr_pairs:
            self.problems.append({
                "description": (
                    f"Corrélation élevée entre {pair['col1']} et {pair['col2']}: "
                    f"{pair['correlation']:.2f}"
                ),
                "severity": "medium",
                "suggestion": "Supprimer une des deux colonnes ou utiliser PCA",
            })

    def _find_high_corr_pairs(self, corr_matrix: pd.DataFrame, columns) -> List[Dict[str, Any]]:
        pairs = []
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                corr_value = corr_matrix.iloc[i, j]
                if corr_value > self.threshold:
                    pairs.append({"col1": columns[i], "col2": columns[j], "correlation": corr_value})
        return pairs

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class CardinalityDetector(BaseDetector):
    """Détecte les colonnes catégorielles à cardinalité trop élevée."""

    HIGH_SEVERITY_THRESHOLD = 100

    def __init__(self, max_categories: int = 50, **kwargs):
        """
        Args:
            max_categories: Nombre maximum de catégories recommandé.
        """
        super().__init__(**kwargs)
        self.max_categories = max_categories
        self.cardinality_stats: Dict[str, int] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.problems = []
        self.cardinality_stats = {}

        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            n_unique = X[col].nunique()
            self.cardinality_stats[col] = n_unique

            if n_unique > self.max_categories:
                severity = "high" if n_unique > self.HIGH_SEVERITY_THRESHOLD else "medium"
                self.problems.append({
                    "column": col,
                    "description": f"{n_unique} catégories (recommandé: < {self.max_categories})",
                    "severity": severity,
                    "suggestion": self._suggest_encoding(n_unique),
                })

    @staticmethod
    def _suggest_encoding(n_unique: int) -> str:
        if n_unique < 10:
            return "One-Hot Encoding"
        if n_unique < 50:
            return "Target Encoding"
        return "Frequency Encoding ou Binary Encoding"

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class DuplicateDetector(BaseDetector):
    """Détecte les lignes dupliquées."""

    HIGH_SEVERITY_THRESHOLD = 100

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.duplicate_count = 0

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.problems = []
        self.duplicate_count = int(X.duplicated().sum())

        if self.duplicate_count > 0:
            severity = "medium" if self.duplicate_count > self.HIGH_SEVERITY_THRESHOLD else "low"
            self.problems.append({
                "description": f"{self.duplicate_count} lignes dupliquées",
                "severity": severity,
                "suggestion": "Supprimer les doublons",
            })

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X