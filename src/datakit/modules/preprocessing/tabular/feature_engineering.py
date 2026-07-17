# preprocessing/tabular/feature_engineering.py
import math
import warnings
from itertools import combinations
from typing import Optional, List, Dict, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from ..base import BasePreprocessor
from ._column_utils import select_columns


class PolynomialFeatureCreator(BasePreprocessor):
    """Crée des features polynomiales, avec garde-fous sur la taille de sortie."""

    def __init__(self,
                 degree: int = 2,
                 columns: Optional[List[str]] = None,
                 interaction_only: bool = False,
                 include_bias: bool = False,
                 max_features: int = 50,
                 max_output_features: int = 5000,
                 **kwargs):
        super().__init__(**kwargs)
        self.degree = degree
        self.columns = columns
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.max_features = max_features
        self.max_output_features = max_output_features

        self.poly = None
        self.feature_names: List[str] = []
        self.columns_to_use: List[str] = []

    def _count_output_features(self, n: int, d: int) -> int:
        """Nombre de features polynomiales produites, sans les générer."""
        if self.interaction_only:
            return sum(math.comb(n, k) for k in range(2, d + 1))
        return math.comb(n + d, d) - 1  # -1 pour exclure la constante

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        cols_to_use = select_columns(X, self.columns, dtype_include=[np.number])

        if len(cols_to_use) > self.max_features:
            raise ValueError(
                f"Too many features ({len(cols_to_use)}) for polynomial creation. "
                f"Max is {self.max_features}. Please select fewer columns or increase max_features."
            )

        n_output = self._count_output_features(len(cols_to_use), self.degree)
        if n_output > self.max_output_features:
            raise ValueError(
                f"Polynomial features would create {n_output} features "
                f"(max is {self.max_output_features}). Please reduce degree or number of columns."
            )

        self.columns_to_use = cols_to_use
        if not cols_to_use:
            return

        self.poly = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )
        self.poly.fit(X[cols_to_use])
        self.feature_names = list(self.poly.get_feature_names_out(cols_to_use))

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns_to_use:
            return X.copy()

        poly_features = self.poly.transform(X[self.columns_to_use])
        poly_df = pd.DataFrame(poly_features, columns=self.feature_names, index=X.index)

        return pd.concat([X.drop(columns=self.columns_to_use), poly_df], axis=1)


class InteractionFeatureCreator(BasePreprocessor):
    """Crée des features de produit (col_a * col_b * ...), avec garde-fou sur la taille de sortie."""

    def __init__(self,
                 columns: Optional[List[str]] = None,
                 max_interactions: int = 2,
                 max_output_features: int = 5000,
                 **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.max_interactions = max_interactions
        self.max_output_features = max_output_features
        self.interaction_combinations: List[Tuple[str, ...]] = []

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        cols_to_use = select_columns(X, self.columns, dtype_include=[np.number])

        combos = [
            combo
            for r in range(2, min(self.max_interactions, len(cols_to_use)) + 1)
            for combo in combinations(cols_to_use, r)
        ]

        if len(combos) > self.max_output_features:
            raise ValueError(
                f"Interaction features would create {len(combos)} columns "
                f"(max is {self.max_output_features}). Reduce max_interactions or number of columns."
            )

        self.interaction_combinations = combos

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for combo in self.interaction_combinations:
            col_name = "*".join(combo)
            X_copy[col_name] = X_copy[list(combo)].prod(axis=1)
        return X_copy


class RatioFeatureCreator(BasePreprocessor):
    """Crée des ratios col_a/col_b pour chaque paire de colonnes numériques."""

    def __init__(self,
                 columns: Optional[List[str]] = None,
                 epsilon: float = 1e-6,
                 max_pairs: int = 100,
                 **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.epsilon = epsilon
        self.max_pairs = max_pairs
        self.column_pairs: List[Tuple[str, str]] = []

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        cols_to_use = select_columns(X, self.columns, dtype_include=[np.number])
        all_pairs = list(combinations(cols_to_use, 2))

        if len(all_pairs) > self.max_pairs:
            warnings.warn(
                f"Too many pairs ({len(all_pairs)}). Limiting to the first {self.max_pairs}. "
                "Consider selecting fewer columns or increasing max_pairs.",
                RuntimeWarning,
            )
            all_pairs = all_pairs[:self.max_pairs]

        self.column_pairs = all_pairs

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col1, col2 in self.column_pairs:
            X_copy[f"{col1}_over_{col2}"] = X_copy[col1] / (X_copy[col2] + self.epsilon)
            X_copy[f"{col2}_over_{col1}"] = X_copy[col2] / (X_copy[col1] + self.epsilon)
        return X_copy


class AggregationFeatureCreator(BasePreprocessor):
    """Crée des features par agrégation groupée (ex: moyenne d'une colonne par groupe)."""

    DEFAULT_AGGREGATIONS = ("mean", "sum", "std", "min", "max", "count")

    def __init__(self,
                 group_column: Optional[str] = None,
                 agg_columns: Optional[List[str]] = None,
                 aggregations: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.group_column = group_column
        self.agg_columns = agg_columns
        self.aggregations = aggregations or list(self.DEFAULT_AGGREGATIONS)

        self.agg_mapping: Dict[str, Dict[str, Dict]] = {}
        self.agg_names: List[str] = []

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        if self.group_column is None or self.group_column not in X.columns:
            raise ValueError(f"Group column '{self.group_column}' not found")

        agg_columns = self.agg_columns
        if agg_columns is None:
            agg_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            if self.group_column in agg_columns:
                agg_columns.remove(self.group_column)

        # Reset : évite l'accumulation si _fit est appelé plusieurs fois sur le même objet.
        self.agg_mapping = {}
        self.agg_names = []

        for col in agg_columns:
            if col not in X.columns:
                continue
            grouped = X.groupby(self.group_column)[col]
            agg_dict = {agg: grouped.agg(agg).to_dict() for agg in self.aggregations}
            self.agg_mapping[col] = agg_dict
            self.agg_names.extend(f"{col}_{agg}" for agg in self.aggregations)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col, agg_dict in self.agg_mapping.items():
            for agg, mapping in agg_dict.items():
                X_copy[f"{col}_{agg}"] = X_copy[self.group_column].map(mapping)
        return X_copy


class DateFeatureCreator(BasePreprocessor):
    """
    Extrait des features (année, mois, jour...) à partir de colonnes de date.
    Seules les colonnes datetime64, ou convertibles en datetime, sont traitées.
    """

    # Chaque builder reçoit la série datetime et retourne la série de la feature.
    _FEATURE_BUILDERS: Dict[str, Callable[[pd.Series], pd.Series]] = {
        "year": lambda s: s.dt.year,
        "month": lambda s: s.dt.month,
        "day": lambda s: s.dt.day,
        "dayofweek": lambda s: s.dt.dayofweek,
        "quarter": lambda s: s.dt.quarter,
        "is_weekend": lambda s: (s.dt.dayofweek >= 5).astype(int),
    }

    def __init__(self,
                 date_columns: Optional[List[str]] = None,
                 create_year: bool = True,
                 create_month: bool = True,
                 create_day: bool = True,
                 create_dayofweek: bool = True,
                 create_quarter: bool = True,
                 create_is_weekend: bool = True,
                 auto_detect: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.date_columns = date_columns
        self.create_year = create_year
        self.create_month = create_month
        self.create_day = create_day
        self.create_dayofweek = create_dayofweek
        self.create_quarter = create_quarter
        self.create_is_weekend = create_is_weekend
        self.auto_detect = auto_detect  # désactivé par défaut pour éviter les faux positifs

        self.columns_to_process: List[str] = []

    @property
    def _active_features(self) -> List[str]:
        flags = {
            "year": self.create_year,
            "month": self.create_month,
            "day": self.create_day,
            "dayofweek": self.create_dayofweek,
            "quarter": self.create_quarter,
            "is_weekend": self.create_is_weekend,
        }
        return [name for name, enabled in flags.items() if enabled]

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        if self.date_columns is not None:
            self.columns_to_process = [c for c in self.date_columns if c in X.columns]
        elif self.auto_detect:
            self.columns_to_process = X.select_dtypes(include=["datetime64"]).columns.tolist()
        else:
            self.columns_to_process = []
            warnings.warn(
                "No date_columns specified and auto_detect is False. "
                "Specify date_columns to process dates.",
                RuntimeWarning,
            )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        active_features = self._active_features

        for col in self.columns_to_process:
            series = self._as_datetime(X_copy[col], col)
            if series is None or series.isna().all():
                continue
            X_copy[col] = series

            for feature_name in active_features:
                builder = self._FEATURE_BUILDERS[feature_name]
                X_copy[f"{col}_{feature_name}"] = builder(series)

        return X_copy

    @staticmethod
    def _as_datetime(series: pd.Series, col_name: str) -> Optional[pd.Series]:
        """Convertit en datetime si nécessaire ; None si la conversion échoue totalement."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return series

        converted = pd.to_datetime(series, errors="coerce")
        if converted.isna().all():
            warnings.warn(
                f"Column '{col_name}' could not be converted to datetime; skipping.",
                RuntimeWarning,
            )
            return None
        return converted