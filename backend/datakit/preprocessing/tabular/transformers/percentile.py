
# preprocessing/tabular/transformers/percentile.py

from typing import Optional, List, Dict

import numpy as np
import pandas as pd


from ...base import BasePreprocessor
from ..utils._column_utils import select_columns


class PercentileTransformer(BasePreprocessor):
    """
    Transforms each value into its percentile rank (0 to 1) relative to
    the distribution learned during fit.

    Note: this functionally overlaps with FeatureScaler(
    method=ScalingMethod.QUANTILE) in scalers.py, which relies on
    sklearn.QuantileTransformer (more robust and natively handles
    out-of-range values). Use this only if the raw percentile rank
    (rather than a normal/uniform distribution) is explicitly desired;
    otherwise, prefer FeatureScaler.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        n_quantiles: int = 1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.n_quantiles = n_quantiles
        self.columns_to_transform: List[str] = []
        self.quantiles: Dict[str, np.ndarray] = {}

    def _fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> None:
        self.columns_to_transform = select_columns(
            X,
            self.columns,
            dtype_include=[np.number]
        )

        self.quantiles = {
            col: np.percentile(
                X[col],
                np.linspace(
                    0,
                    100,
                    self.n_quantiles
                )
            )
            for col in self.columns_to_transform
        }

    def _transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        X_copy = X.copy()

        for col in self.columns_to_transform:
            quantiles = self.quantiles[col]

            X_copy[col] = (
                np.searchsorted(
                    quantiles,
                    X_copy[col]
                )
                / len(quantiles)
            )

        return X_copy
