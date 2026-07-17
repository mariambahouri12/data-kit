# preprocessing/tabular/detectors/missing_value.py

from typing import Dict, Any, Optional

import pandas as pd

from ...base import BaseDetector


class MissingValueDetector(BaseDetector):
    """
    Detects columns with problematic missing values.
    """

    HIGH_SEVERITY_THRESHOLD_PCT = 20.0

    def __init__(
        self,
        threshold: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.threshold = threshold
        self.missing_stats: Dict[str, Any] = {}


    def _fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> None:

        self.problems = []

        self.missing_stats = {
            "total_missing": int(X.isnull().sum().sum()),
            "total_cells": X.size,
            "missing_percentage":
                (X.isnull().sum().sum() / X.size) * 100,
            "columns": {}
        }

        for col in X.columns:

            missing_count = X[col].isnull().sum()
            missing_pct = (
                missing_count / len(X)
            ) * 100

            self.missing_stats["columns"][col] = {
                "missing_count": int(missing_count),
                "missing_percentage": missing_pct
            }

            if missing_pct > self.threshold * 100:

                severity = (
                    "high"
                    if missing_pct > self.HIGH_SEVERITY_THRESHOLD_PCT
                    else "medium"
                )

                self.problems.append(
                    {
                        "column": col,
                        "description":
                            f"{missing_pct:.1f}% missing values",
                        "severity": severity
                    }
                )


    def _transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:

        return X