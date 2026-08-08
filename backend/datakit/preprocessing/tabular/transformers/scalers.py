
# preprocessing/tabular/transformers/scalers.py

from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, QuantileTransformer, PowerTransformer,
)

from ...base import BasePreprocessor
from ..utils._column_utils import select_columns
from ..config import ScalingMethod


class FeatureScaler(BasePreprocessor):
    """Flexible scaler for numerical features.

    Supports: standard, minmax, robust, maxabs, quantile, power
    (Yeo-Johnson or Box-Cox depending on `power_method`).
    """

    def __init__(
        self,
        method: Union[str, ScalingMethod] = ScalingMethod.STANDARD,
        columns: Optional[List[str]] = None,
        with_mean: bool = True,
        with_std: bool = True,
        power_method: str = "yeo-johnson",
        **kwargs
    ):
        """
        Args:
            method: Scaling method.
            columns: Columns to scale (None = all numerical columns).
            with_mean: StandardScaler - center the data.
            with_std: StandardScaler - scale the data.
            power_method: 'yeo-johnson' (supports negative values) or
                'box-cox' (strictly positive values only),
                used only if method == ScalingMethod.POWER.
        """
        super().__init__(**kwargs)
        self.method = ScalingMethod(method) if isinstance(method, str) else method
        self.columns = columns
        self.with_mean = with_mean
        self.with_std = with_std
        self.power_method = power_method

        self.scaler = None
        self.scaler_type: Optional[str] = None
        self.columns_to_scale: List[str] = []

    _SIMPLE_SCALERS = {
        ScalingMethod.MINMAX: MinMaxScaler,
        ScalingMethod.ROBUST: RobustScaler,
        ScalingMethod.MAXABS: MaxAbsScaler,
    }

    def _build_scaler(self):
        if self.method == ScalingMethod.STANDARD:
            return StandardScaler(
                with_mean=self.with_mean,
                with_std=self.with_std
            )

        if self.method == ScalingMethod.QUANTILE:
            return QuantileTransformer(output_distribution="normal")

        if self.method == ScalingMethod.POWER:
            return PowerTransformer(method=self.power_method)

        if self.method in self._SIMPLE_SCALERS:
            return self._SIMPLE_SCALERS[self.method]()

        raise ValueError(
            f"Unsupported scaling method: {self.method}"
        )

    def _fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> None:
        self.columns_to_scale = select_columns(
            X,
            self.columns,
            dtype_include=[np.number]
        )

        if not self.columns_to_scale:
            return

        self.scaler = self._build_scaler()
        self.scaler.fit(X[self.columns_to_scale])
        self.scaler_type = type(self.scaler).__name__

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        if self.columns_to_scale:
            X_copy[self.columns_to_scale] = self.scaler.transform(
                X_copy[self.columns_to_scale]
            )

        return X_copy

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the transformation (not supported by QuantileTransformer
        with certain configurations, nor by PowerTransformer for columns
        outside the original domain)."""
        X_copy = X.copy()

        if self.columns_to_scale:
            X_copy[self.columns_to_scale] = self.scaler.inverse_transform(
                X_copy[self.columns_to_scale]
            )

        return X_copy

    def get_scale_params(self) -> Dict[str, Any]:
        """Return the learned scaling parameters. The content depends on
        the scaler type (e.g., mean_/scale_ for StandardScaler,
        data_min_/data_max_ for MinMaxScaler)."""
        if self.scaler is None:
            return {}

        params = {}

        for attr in (
            "mean_",
            "scale_",
            "center_",
            "data_min_",
            "data_max_"
        ):
            if hasattr(self.scaler, attr):
                params[attr.rstrip("_")] = getattr(
                    self.scaler,
                    attr
                ).tolist()

        return params

