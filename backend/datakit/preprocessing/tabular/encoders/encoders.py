
# preprocessing/tabular/encoders.py

from typing import Optional, List, Dict, Any, Union

import pandas as pd

from ...base import BasePreprocessor
from ..utils._column_utils import select_columns
from ..config import EncodingMethod
from . import _encoding_utils as _eu


class CategoricalEncoder(BasePreprocessor):
    """
    Flexible encoder for categorical variables.
    Supports: One-Hot, Label, Target, Frequency, Binary, CatBoost, Hash, Ordinal.

    Columns whose number of categories exceeds `max_categories` are
    automatically switched to frequency encoding, regardless of the
    selected method, to avoid dimensionality explosion (e.g., one-hot
    encoding on an ID with 50,000 values).
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

    def __init__(
        self,
        method: Union[str, EncodingMethod] = EncodingMethod.ONE_HOT,
        columns: Optional[List[str]] = None,
        max_categories: int = 50,
        min_frequency: float = 0.01,
        handle_unknown: str = "ignore",
        sparse: bool = True,
        **kwargs
    ):
        """
        Args:
            method: Encoding method.
            columns: Columns to encode (None = all categorical columns).
            max_categories: Maximum number of categories before falling
                back to frequency encoding.
            min_frequency: Minimum frequency required to keep a category
                separate (passed to sklearn.OneHotEncoder; rarer categories
                are grouped as 'infrequent').
            handle_unknown: 'ignore' or 'error' for unknown categories
                during transform.
            sparse: Use a sparse matrix for One-Hot encoding (RAM savings).
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
        self.fallback_columns: List[str] = []

    # -- Fit -----------------------------------------------------------------

    def _fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> None:
        candidate_cols = select_columns(
            X,
            self.columns,
            dtype_include=["object", "category"]
        )

        self.columns_to_encode, self.fallback_columns = (
            _eu.split_by_cardinality(
                self,
                X,
                candidate_cols
            )
        )

        if self.fallback_columns:
            _eu.fit_frequency(
                self,
                X,
                self.fallback_columns
            )

        if (
            self.method == EncodingMethod.NONE
            or not self.columns_to_encode
        ):
            return

        self._FIT_METHODS[self.method](
            self,
            X,
            y
        )

    # -- Transform -------------------------------------------------------------

    def _transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        if self.method == EncodingMethod.ONE_HOT:
            X_copy = _eu.transform_onehot(
                self,
                X
            )

        elif self.method == EncodingMethod.ORDINAL:
            X_copy = _eu.transform_ordinal(
                self,
                X
            )

        else:
            X_copy = _eu.transform_other(
                self,
                X
            )

        return _eu.apply_frequency_fallback(
            self,
            X_copy
        )

    # -- Introspection -----------------------------------------------------

    def get_feature_names(self) -> List[str]:
        """Names of the columns produced after encoding."""
        if self.method == EncodingMethod.ONE_HOT:
            return (
                list(self.column_names)
                + list(self.fallback_columns)
            )

        if self.method == EncodingMethod.BINARY:
            binary_names = [
                f"{col}_bit_{i}"
                for col in self.columns_to_encode
                for i in range(
                    self.mapping[col]["n_bits"]
                )
            ]

            return (
                binary_names
                + list(self.fallback_columns)
            )

        return (
            list(self.columns_to_encode)
            + list(self.fallback_columns)
        )

