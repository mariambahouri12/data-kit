"""
Dataset context generation.
"""

from typing import Any

import pandas as pd

from datakit.preprocessing.tabular.detectors import (
    CardinalityDetector,
    CorrelationDetector,
    DuplicateDetector,
    MissingValueDetector,
    OutlierDetector,
)


class DatasetContextBuilder:
    """Build an LLM-friendly dataset context."""

    def __init__(
        self,
        missing_threshold: float = 0.05,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        correlation_threshold: float = 0.8,
        max_categories: int = 50,
    ) -> None:
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.correlation_threshold = correlation_threshold
        self.max_categories = max_categories

    def build(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str = "unknown",
    ) -> dict[str, Any]:

        if dataframe is None or dataframe.empty:
            return {
                "dataset_name": dataset_name,
                "shape": {"rows": 0, "columns": 0},
                "detected_problems": [],
                "detectors": {},
            }

        detectors = {
            "missing_values": MissingValueDetector(
                threshold=self.missing_threshold
            ),
            "outliers": OutlierDetector(
                method=self.outlier_method,
                threshold=self.outlier_threshold,
            ),
            "correlation": CorrelationDetector(
                threshold=self.correlation_threshold
            ),
            "cardinality": CardinalityDetector(
                max_categories=self.max_categories
            ),
            "duplicates": DuplicateDetector(),
        }

        for detector in detectors.values():
            detector.fit(dataframe)

        return {
            "dataset_name": dataset_name,
            "shape": {
                "rows": int(dataframe.shape[0]),
                "columns": int(dataframe.shape[1]),
            },
            "detected_problems": self._collect_problems(detectors),
            "detectors": {
                "missing_values": self._missing_context(
                    detectors["missing_values"]
                ),
                "outliers": self._outlier_context(
                    detectors["outliers"]
                ),
                "correlation": self._correlation_context(
                    detectors["correlation"]
                ),
                "cardinality": self._cardinality_context(
                    detectors["cardinality"]
                ),
                "duplicates": self._duplicate_context(
                    detectors["duplicates"]
                ),
            },
        }

    @staticmethod
    def _collect_problems(
        detectors: dict[str, Any],
    ) -> list[dict[str, Any]]:
        problems = []

        for name, detector in detectors.items():
            for problem in getattr(detector, "problems", []):
                problems.append({"detector": name, **problem})

        return problems

    @staticmethod
    def _missing_context(
        detector: MissingValueDetector,
    ) -> dict[str, Any]:

        stats = detector.missing_stats

        return {
            "total_missing": int(stats.get("total_missing", 0)),
            "total_cells": int(stats.get("total_cells", 0)),
            "missing_percentage": round(
                float(stats.get("missing_percentage", 0)),
                2,
            ),
            "columns": stats.get("columns", {}),
        }

    @staticmethod
    def _outlier_context(
        detector: OutlierDetector,
    ) -> dict[str, Any]:

        method = (
            detector.method.value
            if hasattr(detector.method, "value")
            else str(detector.method)
        )

        return {
            "method": method,
            "threshold": detector.threshold,
            "columns": detector.outlier_stats,
        }

    @staticmethod
    def _correlation_context(
        detector: CorrelationDetector,
    ) -> dict[str, Any]:

        correlations = detector.correlations

        pairs = correlations.get("high_corr_pairs", [])

        return {
            "threshold": detector.threshold,
            "high_correlation_pairs": pairs,
            "high_correlation_count": len(pairs),
        }

    @staticmethod
    def _cardinality_context(
        detector: CardinalityDetector,
    ) -> dict[str, Any]:

        return {
            "max_categories": detector.max_categories,
            "columns": detector.cardinality_stats,
        }

    @staticmethod
    def _duplicate_context(
        detector: DuplicateDetector,
    ) -> dict[str, Any]:

        return {"duplicate_count": detector.duplicate_count}