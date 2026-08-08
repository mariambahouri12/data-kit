
import warnings
from typing import Callable, Dict, List, Optional

import pandas as pd

from ...base import BasePreprocessor


class DateFeatureCreator(BasePreprocessor):
    """
    Extracts features (year, month, day...) from date columns.
    Only datetime64 columns, or columns convertible to datetime, are processed.
    """

    # Each builder receives the datetime series and returns the feature series.
    _FEATURE_BUILDERS: Dict[str, Callable[[pd.Series], pd.Series]] = {
        "year": lambda s: s.dt.year,
        "month": lambda s: s.dt.month,
        "day": lambda s: s.dt.day,
        "dayofweek": lambda s: s.dt.dayofweek,
        "quarter": lambda s: s.dt.quarter,
        "is_weekend": lambda s: (s.dt.dayofweek >= 5).astype(int),
    }

    def __init__(
        self,
        date_columns: Optional[List[str]] = None,
        create_year: bool = True,
        create_month: bool = True,
        create_day: bool = True,
        create_dayofweek: bool = True,
        create_quarter: bool = True,
        create_is_weekend: bool = True,
        auto_detect: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.date_columns = date_columns
        self.create_year = create_year
        self.create_month = create_month
        self.create_day = create_day
        self.create_dayofweek = create_dayofweek
        self.create_quarter = create_quarter
        self.create_is_weekend = create_is_weekend
        self.auto_detect = auto_detect  # disabled by default to avoid false positives

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

        return [
            name
            for name, enabled in flags.items()
            if enabled
        ]

    def _fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> None:
        if self.date_columns is not None:
            self.columns_to_process = [
                c
                for c in self.date_columns
                if c in X.columns
            ]

        elif self.auto_detect:
            self.columns_to_process = (
                X.select_dtypes(
                    include=["datetime64"]
                ).columns.tolist()
            )

        else:
            self.columns_to_process = []

            warnings.warn(
                "No date_columns specified and auto_detect is False. "
                "Specify date_columns to process dates.",
                RuntimeWarning,
            )

    def _transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        X_copy = X.copy()
        active_features = self._active_features

        for col in self.columns_to_process:
            series = self._as_datetime(
                X_copy[col],
                col
            )

            if series is None or series.isna().all():
                continue

            X_copy[col] = series

            for feature_name in active_features:
                builder = self._FEATURE_BUILDERS[feature_name]
                X_copy[f"{col}_{feature_name}"] = builder(series)

        return X_copy

    @staticmethod
    def _as_datetime(
        series: pd.Series,
        col_name: str
    ) -> Optional[pd.Series]:
        """Converts to datetime if necessary; returns None if conversion fails completely."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return series

        converted = pd.to_datetime(
            series,
            errors="coerce"
        )

        if converted.isna().all():
            warnings.warn(
                f"Column '{col_name}' could not be converted to datetime; skipping.",
                RuntimeWarning,
            )
            return None

        return converted

