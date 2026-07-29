"""
Data validation orchestration module.

This module coordinates different data quality detectors
and aggregates their results.
"""

from typing import Dict, Any, Optional

import pandas as pd

from ..preprocessing.tabular.detectors.missing_value import MissingValueDetector
from ..preprocessing.tabular.detectors.duplicate import DuplicateDetector
from ..preprocessing.tabular.detectors.outlier import OutlierDetector
from ..preprocessing.tabular.detectors.correlation import CorrelationDetector
from ..preprocessing.tabular.detectors.cardinality import CardinalityDetector


class DataValidator:
    """
    Runs all data quality detectors and aggregates validation results.
    """

    def __init__(self):
        """
        Initialize all available detectors.
        """

        self.detectors = [
            MissingValueDetector(),
            DuplicateDetector(),
            OutlierDetector(),
            CorrelationDetector(),
            CardinalityDetector(),
        ]

        self.validation_results = {}


    def validate(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Validate a DataFrame using all registered detectors.

        Args:
            X: DataFrame to validate.
            y: Optional target variable.

        Returns:
            Dictionary containing validation results.
        """

        results = {
            "is_valid": True,
            "detectors": {},
            "problems": [],
            "problems_by_column": {},
            "summary": {
                "total_problems": 0,
                "columns_with_issues": 0,
            },
        }


        for detector in self.detectors:

            detector_name = detector.__class__.__name__

            try:
                # Run detection
                detector.fit(X, y)

                problems = detector.problems


                # Store detector results
                results["detectors"][detector_name] = {
                    "problems": problems,
                    "count": len(problems),
                }


                # Add global problems list
                results["problems"].extend(
                    problems
                )
                
                # Group problems by column
                for problem in problems:
                    column = problem.get("column")
                    if column:
                        if column not in results["problems_by_column"]:
                            results["problems_by_column"][column] = []
                        results["problems_by_column"][column].append(problem)


            except Exception as error:

                results["detectors"][detector_name] = {
                    "error": str(error),
                    "problems": []
                }


                results["problems"].append(
                    {
                        "detector": detector_name,
                        "description": str(error),
                    }
                )


        # Update final status
        results["summary"]["total_problems"] = len(
            results["problems"]
        )
        
        # Count columns with issues
        results["summary"]["columns_with_issues"] = len(results["problems_by_column"].keys())


        if results["summary"]["total_problems"] > 0:
            results["is_valid"] = False


        self.validation_results = results

        return results