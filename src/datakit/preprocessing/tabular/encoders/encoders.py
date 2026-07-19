# preprocessing/tabular/encoders.py
from typing import Optional, List, Dict, Any, Union

import pandas as pd

from ...base import BasePreprocessor
from ..utils._column_utils import select_columns
from ..config import EncodingMethod
from . import _encoding_utils as _eu


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
        EncodingMethod.ONE_HOT: _eu.fit_onehot,
        EncodingMethod.LABEL: _eu.fit_label,
        EncodingMethod.TARGET: _eu.fit_target,
        EncodingMethod.FREQUENCY: _eu.fit_frequency,
        EncodingMethod.BINARY: _eu.fit_binary,
        EncodingMethod.CATBOOST: _eu.fit_catboost,
        EncodingMethod.HASH: _eu.fit_hash,
        EncodingMethod.ORDINAL: _eu.fit_ordinal,
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

        self.columns_to_encode, self.fallback_columns = _eu.split_by_cardinality(self, X, candidate_cols)

        if self.fallback_columns:
            _eu.fit_frequency(self, X, self.fallback_columns)

        if self.method == EncodingMethod.NONE or not self.columns_to_encode:
            return

        self._FIT_METHODS[self.method](self, X, y)

    # -- Transform -------------------------------------------------------------

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.method == EncodingMethod.ONE_HOT:
            X_copy = _eu.transform_onehot(self, X)
        elif self.method == EncodingMethod.ORDINAL:
            X_copy = _eu.transform_ordinal(self, X)
        else:
            X_copy = _eu.transform_other(self, X)

        return _eu.apply_frequency_fallback(self, X_copy)

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