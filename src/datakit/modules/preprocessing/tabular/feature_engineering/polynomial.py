import math
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from ...base import BasePreprocessor
from .._column_utils import select_columns

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