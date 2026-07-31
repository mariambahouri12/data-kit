# preprocessing/tabular/detectors/outlier.py

import numpy as np
import pandas as pd

from typing import Dict, Optional, Union

from ...base import BaseDetector
from ..config import OutlierMethod


class OutlierDetector(BaseDetector):
    """Detect the proportion of outliers per numeric column."""

    def __init__(self, method: Union[str, OutlierMethod] = OutlierMethod.IQR,
                 threshold: float = 1.5, **kwargs):
        """
        Args:
            method: IQR ou ZSCORE.
            threshold: 1.5 typical for IQR, 3 for z-score.
        """
        super().__init__(**kwargs)
        self.method = OutlierMethod(method) if isinstance(method, str) else method
        self.threshold = threshold
        self.outlier_stats: Dict[str, Dict[str, float]] = {}

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:

        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_data = X[col].dropna()
            if col_data.empty:
                continue

            n_outliers = self._count_outliers(X[col], col_data)
            if n_outliers is None:
                continue

            outlier_pct = (n_outliers / len(col_data)) * 100
            self.outlier_stats[col] = {"n_outliers": n_outliers, "percentage": outlier_pct}

            if n_outliers > 0:
                self.problems.append({
                    "column": col,
                    "description": f"{n_outliers} outliers ({outlier_pct:.1f}%)",
                })

    def _count_outliers(self, full_column: pd.Series, non_null_data: pd.Series) -> Optional[int]:
        """Returns the number of outliers, or None if the calculation isn't possible (e.g., std=0)."""
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

        raise ValueError(f"Outlier detection method not supported: {self.method}")

 