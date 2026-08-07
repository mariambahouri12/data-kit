# preprocessing/tabular/detectors/outlier.py

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from ...base import BaseDetector
from ..config import OutlierMethod


class OutlierDetector(BaseDetector):
    """
    Detect the proportion of outliers per numeric column.

    Supported methods:
    - IQR
    - ZSCORE
    """

    def __init__(
        self,
        method: Union[str, OutlierMethod] = OutlierMethod.IQR,
        threshold: float = 1.5,
        **kwargs
    ):
        """
        Args:
            method:
                Detection method (IQR or ZSCORE).

            threshold:
                - IQR: commonly 1.5
                - ZSCORE: commonly 3
        """

        super().__init__(**kwargs)

        try:
            self.method = (
                OutlierMethod(method)
                if isinstance(method, str)
                else method
            )
        except ValueError:
            raise ValueError(
                f"Unsupported outlier method: {method}"
            )

        self.threshold = threshold

        self.outlier_stats: Dict[
            str,
            Dict[str, float]
        ] = {}

    def _fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> None:
        """
        Detect outliers in numerical columns.
        """

        # Reset previous results
        self.outlier_stats = {}
        self.problems = []

        numeric_cols = X.select_dtypes(
            include=[np.number]
        ).columns

        for col in numeric_cols:

            col_data = X[col].dropna()

            # Ignore empty columns
            if col_data.empty:
                continue

            n_outliers = self._count_outliers(
                X[col],
                col_data
            )

            # Cannot compute outliers
            if n_outliers is None:
                continue

            outlier_percentage = (
                n_outliers / len(col_data)
            ) * 100

            self.outlier_stats[col] = {
                "n_outliers": n_outliers,
                "percentage": outlier_percentage
            }

            if n_outliers > 0:
                self.problems.append(
                    {
                        "column": col,
                        "description":
                            f"{n_outliers} outliers "
                            f"({outlier_percentage:.1f}%)"
                    }
                )

    def _count_outliers(
        self,
        full_column: pd.Series,
        non_null_data: pd.Series
    ) -> Optional[int]:
        """
        Return number of detected outliers.

        Returns:
            int:
                Number of outliers.

            None:
                When calculation is impossible.
        """

        if self.method == OutlierMethod.IQR:

            q1 = non_null_data.quantile(0.25)
            q3 = non_null_data.quantile(0.75)

            iqr = q3 - q1

            lower_bound = (
                q1 - self.threshold * iqr
            )

            upper_bound = (
                q3 + self.threshold * iqr
            )

            return int(
                (
                    (full_column < lower_bound)
                    |
                    (full_column > upper_bound)
                ).sum()
            )


        if self.method == OutlierMethod.ZSCORE:

            mean = non_null_data.mean()
            std = non_null_data.std()

            # Constant column
            if std == 0:
                return None

            z_scores = np.abs(
                (full_column - mean) / std
            )

            return int(
                (z_scores > self.threshold).sum()
            )


        raise ValueError(
            f"Outlier detection method not supported: {self.method}"
        )