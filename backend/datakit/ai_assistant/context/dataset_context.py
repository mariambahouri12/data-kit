"""
Dataset context generation module.

Builds a compact representation of the dataset
using the tabular preprocessing detectors.
"""

from typing import Dict, Any, Optional

import pandas as pd

from datakit.preprocessing.tabular.detectors import (
    MissingValueDetector,
    OutlierDetector,
    CorrelationDetector,
    CardinalityDetector,
    DuplicateDetector,
)


class DatasetContextBuilder:
    """
    Generate dataset information usable by the AI assistant.

    The builder delegates data-quality analysis to the existing
    preprocessing detectors instead of duplicating detection logic.
    """

    def __init__(
        self,
        missing_threshold: float = 0.05,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        correlation_threshold: float = 0.8,
        max_categories: int = 50,
    ):
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.correlation_threshold = correlation_threshold
        self.max_categories = max_categories

    def build(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Build complete dataset context.

        The context contains:
        - dataset name
        - number of rows
        - number of columns
        - detected data-quality problems
        - detailed detector results
        """

        if dataframe is None or dataframe.empty:
            return {
                "dataset_name": dataset_name,
                "shape": {
                    "rows": 0,
                    "columns": 0,
                },
                "detected_problems": [],
                "detectors": {},
            }

        # ---------------------------------------------------------
        # Run detectors
        # ---------------------------------------------------------

        detectors = {
            "missing_values": MissingValueDetector(
                threshold=self.missing_threshold
            ),

            "outliers": OutlierDetector(
                method=self.outlier_method,
                threshold=self.outlier_threshold
            ),

            "correlation": CorrelationDetector(
                threshold=self.correlation_threshold
            ),

            "cardinality": CardinalityDetector(
                max_categories=self.max_categories
            ),

            "duplicates": DuplicateDetector(),
        }

        # Execute detection
        for detector in detectors.values():
            detector.fit(dataframe)

        # ---------------------------------------------------------
        # Build context
        # ---------------------------------------------------------

        context = {
            "dataset_name": dataset_name,

            "shape": {
                "rows": int(dataframe.shape[0]),
                "columns": int(dataframe.shape[1]),
            },

            "detected_problems": self._collect_problems(
                detectors
            ),

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

        return context

    # =============================================================
    # Problems
    # =============================================================

    def _collect_problems(
        self,
        detectors: Dict[str, Any],
    ) -> list:
        """
        Collect all problems detected by all detectors.
        """

        problems = []

        for detector_name, detector in detectors.items():

            for problem in getattr(
                detector,
                "problems",
                []
            ):
                problems.append(
                    {
                        "detector": detector_name,
                        **problem,
                    }
                )

        return problems

    # =============================================================
    # Missing values
    # =============================================================

    def _missing_context(
        self,
        detector: MissingValueDetector,
    ) -> Dict[str, Any]:
        """
        Extract JSON-friendly missing-value information.
        """

        stats = detector.missing_stats

        return {
            "total_missing": int(
                stats.get("total_missing", 0)
            ),

            "total_cells": int(
                stats.get("total_cells", 0)
            ),

            "missing_percentage": round(
                float(
                    stats.get(
                        "missing_percentage",
                        0
                    )
                ),
                2
            ),

            "columns": stats.get(
                "columns",
                {}
            ),
        }

    # =============================================================
    # Outliers
    # =============================================================

    def _outlier_context(
        self,
        detector: OutlierDetector,
    ) -> Dict[str, Any]:
        """
        Extract outlier statistics.
        """

        return {
            "method": detector.method.value
            if hasattr(detector.method, "value")
            else str(detector.method),

            "threshold": detector.threshold,

            "columns": detector.outlier_stats,
        }

    # =============================================================
    # Correlation
    # =============================================================

    def _correlation_context(
        self,
        detector: CorrelationDetector,
    ) -> Dict[str, Any]:
        """
        Extract correlation information.

        The full correlation matrix is intentionally not included
        because it is large and not necessary for the LLM context.
        """

        correlations = detector.correlations

        return {
            "threshold": detector.threshold,

            "high_correlation_pairs": correlations.get(
                "high_corr_pairs",
                []
            ),

            "high_correlation_count": len(
                correlations.get(
                    "high_corr_pairs",
                    []
                )
            ),
        }

    # =============================================================
    # Cardinality
    # =============================================================

    def _cardinality_context(
        self,
        detector: CardinalityDetector,
    ) -> Dict[str, Any]:
        """
        Extract categorical cardinality information.
        """

        return {
            "max_categories": detector.max_categories,

            "columns": detector.cardinality_stats,
        }

    # =============================================================
    # Duplicates
    # =============================================================

    def _duplicate_context(
        self,
        detector: DuplicateDetector,
    ) -> Dict[str, Any]:
        """
        Extract duplicate-row information.
        """

        return {
            "duplicate_count": detector.duplicate_count,
        }