# preprocessing/tabular/encoders.py
import hashlib
import warnings
from typing import Optional, List, Dict, Any, Union

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

from ..base import BasePreprocessor
from ._column_utils import select_columns
from .config import EncodingMethod


class CategoricalEncoder(BasePreprocessor):
    """
    Encodeur flexible pour variables catégorielles.
    Supporte : One-Hot, Label, Target, Frequency, Binary, CatBoost, Hash, Ordinal.

    Les colonnes dont le nombre de catégories dépasse `max_categories` sont
    automatiquement basculées en frequency encoding, quelle que soit la
    méthode choisie, pour éviter une explosion dimensionnelle (ex: one-hot
    sur un ID à 50 000 valeurs).
    """

    _FIT_METHODS = {
        EncodingMethod.ONE_HOT: "_fit_onehot",
        EncodingMethod.LABEL: "_fit_label",
        EncodingMethod.TARGET: "_fit_target",
        EncodingMethod.FREQUENCY: "_fit_frequency",
        EncodingMethod.BINARY: "_fit_binary",
        EncodingMethod.CATBOOST: "_fit_catboost",
        EncodingMethod.HASH: "_fit_hash",
        EncodingMethod.ORDINAL: "_fit_ordinal",
    }

    def __init__(self,
                 method: Union[str, EncodingMethod] = EncodingMethod.ONE_HOT,
                 columns: Optional[List[str]] = None,
                 max_categories: int = 50,
                 min_frequency: float = 0.01,
                 handle_unknown: str = "ignore",
                 sparse: bool = True,
                 **kwargs):
        """
        Args:
            method: Méthode d'encodage.
            columns: Colonnes à encoder (None = toutes les catégorielles).
            max_categories: Nombre max de catégories avant repli en frequency encoding.
            min_frequency: Fréquence minimum pour garder une catégorie séparée
                (transmis à sklearn.OneHotEncoder ; les catégories plus rares
                sont regroupées en 'infrequent').
            handle_unknown: 'ignore' ou 'error' pour les catégories inconnues au transform.
            sparse: Utiliser une sparse matrix pour One-Hot (économie de RAM).
        """
        super().__init__(**kwargs)
        self.method = EncodingMethod(method) if isinstance(method, str) else method
        self.columns = columns
        self.max_categories = max_categories
        self.min_frequency = min_frequency
        self.handle_unknown = handle_unknown
        self.sparse = sparse

        self.encoder = None
        self.mapping: Dict[str, Dict[str, Any]] = {}
        self.column_names: List[str] = []
        self.columns_to_encode: List[str] = []
        self.fallback_columns: List[str] = []  # colonnes trop cardinales -> frequency encoding forcé

    # -- Fit -----------------------------------------------------------------

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        candidate_cols = select_columns(X, self.columns, dtype_include=["object", "category"])

        self.columns_to_encode, self.fallback_columns = self._split_by_cardinality(X, candidate_cols)

        if self.fallback_columns:
            self._fit_frequency(X, self.fallback_columns)

        if self.method == EncodingMethod.NONE or not self.columns_to_encode:
            return

        fit_method_name = self._FIT_METHODS[self.method]
        getattr(self, fit_method_name)(X, y)

    def _split_by_cardinality(self, X: pd.DataFrame, columns: List[str]) -> tuple:
        """Sépare les colonnes en (encodables normalement, trop cardinales)."""
        normal, fallback = [], []
        for col in columns:
            n_unique = X[col].nunique()
            if n_unique <= self.max_categories:
                normal.append(col)
            else:
                fallback.append(col)
                warnings.warn(
                    f"Column {col} has {n_unique} categories (> max_categories={self.max_categories}), "
                    f"falling back to frequency encoding.",
                    RuntimeWarning,
                )
        return normal, fallback

    def _fit_onehot(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.encoder = OneHotEncoder(
            handle_unknown=self.handle_unknown,
            sparse_output=self.sparse,
            drop="if_binary",
            min_frequency=self.min_frequency,
        )
        self.encoder.fit(X[self.columns_to_encode])
        self.column_names = list(self.encoder.get_feature_names_out(self.columns_to_encode))

    def _fit_label(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Note : LabelEncoder n'a pas de notion de catégorie inconnue ;
        les valeurs inédites sont gérées manuellement au transform (fallback -1)."""
        self.encoder = {}
        for col in self.columns_to_encode:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoder[col] = le

    def _fit_target(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        WARNING: fuite de données si utilisé sur tout le dataset sans CV.
        Utiliser un KFold target encoding en production.
        """
        target = self._resolve_target(y)
        global_mean = target.mean()
        tmp = X.assign(_target_=target.values)

        for col in self.columns_to_encode:
            group_sizes = tmp.groupby(col).size()
            group_means = tmp.groupby(col)["_target_"].mean()
            smoothing = group_sizes / (group_sizes + 10)
            smoothed = smoothing * group_means + (1 - smoothing) * global_mean
            self.mapping[col] = {"encoding": smoothed.to_dict()}

    def _fit_frequency(self, X: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        for col in (columns if columns is not None else self.columns_to_encode):
            self.mapping[col] = {"encoding": X[col].value_counts(normalize=True).to_dict()}

    def _fit_binary(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        for col in self.columns_to_encode:
            categories = X[col].unique()
            n_bits = max(1, len(categories).bit_length())
            binary_mapping = {
                cat: [int(b) for b in format(i, f"0{n_bits}b")]
                for i, cat in enumerate(categories)
            }
            self.mapping[col] = {"encoding": binary_mapping, "n_bits": n_bits}

    def _fit_catboost(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        target = self._resolve_target(y)
        global_mean = target.mean()

        for col in self.columns_to_encode:
            group_sizes = X.groupby(col).size()
            group_sums = target.groupby(X[col]).sum()
            cat_means = ((group_sums + global_mean) / (group_sizes + 1)).to_dict()
            self.mapping[col] = {"encoding": cat_means}

    def _fit_hash(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Hash MD5 tronqué pour reproductibilité inter-run (contrairement à hash())."""
        for col in self.columns_to_encode:
            hash_mapping = {
                cat: int(hashlib.md5(str(cat).encode()).hexdigest(), 16) % 1_000_000
                for cat in X[col].unique()
            }
            self.mapping[col] = {"encoding": hash_mapping}

    def _fit_ordinal(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.encoder.fit(X[self.columns_to_encode])

    def _resolve_target(self, y: Optional[pd.Series]) -> pd.Series:
        if y is None:
            raise ValueError(f"{self.method} encoder requires a target variable (y).")
        return y

    # -- Transform -------------------------------------------------------------

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.method == EncodingMethod.ONE_HOT:
            X_copy = self._transform_onehot(X)
        elif self.method == EncodingMethod.ORDINAL:
            X_copy = self._transform_ordinal(X)
        else:
            X_copy = self._transform_other(X)

        return self._apply_frequency_fallback(X_copy)

    def _apply_frequency_fallback(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applique le frequency encoding aux colonnes trop cardinales pour la méthode choisie."""
        X_copy = X.copy()
        for col in self.fallback_columns:
            encoding_map = self.mapping[col]["encoding"]
            X_copy[col] = X_copy[col].map(encoding_map).fillna(0)
        return X_copy

    def _transform_onehot(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns_to_encode:
            return X.copy()

        encoded = self.encoder.transform(X[self.columns_to_encode])
        if self.sparse:
            encoded = encoded.toarray()

        encoded_df = pd.DataFrame(encoded, columns=self.column_names, index=X.index)
        return pd.concat([X.drop(columns=self.columns_to_encode), encoded_df], axis=1)

    def _transform_ordinal(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if self.columns_to_encode and self.encoder is not None:
            X_copy[self.columns_to_encode] = self.encoder.transform(X_copy[self.columns_to_encode])
        return X_copy

    def _transform_other(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        for col in self.columns_to_encode:
            if col not in X_copy.columns:
                continue

            if self.method == EncodingMethod.LABEL:
                X_copy[col] = self._transform_label_column(X_copy[col])
            elif self.method == EncodingMethod.BINARY:
                X_copy = self._transform_binary_column(X_copy, col)
            else:  # TARGET, FREQUENCY, CATBOOST, HASH partagent la même logique de mapping
                X_copy[col] = self._transform_mapped_column(X_copy[col], col)

        return X_copy

    def _transform_label_column(self, column: pd.Series) -> pd.Series:
        le: LabelEncoder = self.encoder[column.name]
        known_classes = set(le.classes_)
        return column.astype(str).apply(
            lambda x: le.transform([x])[0] if x in known_classes else -1
        )

    def _transform_mapped_column(self, column: pd.Series, col_name: str) -> pd.Series:
        encoding_map = self.mapping[col_name]["encoding"]
        result = column.map(encoding_map)

        if result.isna().any():
            if self.handle_unknown == "ignore":
                result = result.fillna(0)
            else:
                raise ValueError(f"Unknown categories found in column {col_name}")

        return result

    def _transform_binary_column(self, X: pd.DataFrame, col_name: str) -> pd.DataFrame:
        n_bits = self.mapping[col_name]["n_bits"]
        encoding_map = self.mapping[col_name]["encoding"]
        default_bits = [0] * n_bits

        for i in range(n_bits):
            X[f"{col_name}_bit_{i}"] = X[col_name].apply(
                lambda x, i=i: encoding_map.get(x, default_bits)[i]
            )

        return X.drop(columns=[col_name])

    # -- Introspection -----------------------------------------------------

    def get_feature_names(self) -> List[str]:
        """Noms des colonnes produites après encodage."""
        if self.method == EncodingMethod.ONE_HOT:
            return list(self.column_names) + list(self.fallback_columns)

        if self.method == EncodingMethod.BINARY:
            binary_names = [
                f"{col}_bit_{i}"
                for col in self.columns_to_encode
                for i in range(self.mapping[col]["n_bits"])
            ]
            return binary_names + list(self.fallback_columns)

        return list(self.columns_to_encode) + list(self.fallback_columns)