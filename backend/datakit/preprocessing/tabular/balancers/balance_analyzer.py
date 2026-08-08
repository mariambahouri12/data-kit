
# tabular/balancers/balance_analyser.py
from typing import Dict, Any

import pandas as pd

from ..config import BalancingMethod


class ImbalanceAnalyzer:
    """
    Analyzes the class distribution of a target and suggests an appropriate
    rebalancing method. Pure diagnostic tool: does not modify data
    and does not require an instantiated ClassBalancer.
    """

    # (exclusive ratio threshold, severity, [(method, reason), ...] by priority)
    _RULES = (
        (2, "low", [(BalancingMethod.NONE, "Classes are already balanced")]),
        (5, "medium", [
            (BalancingMethod.RANDOM_OVER, "Moderate imbalance, simple over-sampling"),
            (BalancingMethod.SMOTE, "Moderate imbalance, SMOTE provides better results"),
        ]),
        (10, "high", [
            (BalancingMethod.SMOTE, "Significant imbalance, SMOTE is recommended"),
            (BalancingMethod.ADASYN, "Alternative to SMOTE for difficult cases"),
        ]),
        (float("inf"), "high", [
            (BalancingMethod.ADASYN, "Severe imbalance, ADASYN is recommended"),
            (BalancingMethod.SMOTE, "Alternative for severe imbalance"),
        ]),
    )

    @staticmethod
    def get_class_distribution(y: pd.Series) -> Dict[str, Any]:
        counts = y.value_counts()
        min_class, max_class = counts.min(), counts.max()
        return {
            "n_classes": len(counts),
            "counts": counts.to_dict(),
            "percentages": (counts / len(y) * 100).to_dict(),
            "imbalance_ratio": max_class / min_class if min_class > 0 else float("inf"),
        }

    @classmethod
    def suggest_method(cls, y: pd.Series) -> Dict[str, Any]:
        distribution = cls.get_class_distribution(y)
        ratio = distribution["imbalance_ratio"]

        for threshold, severity, methods in cls._RULES:
            if ratio < threshold:
                suggestions = [
                    {"method": method, "reason": reason, "priority": i + 1}
                    for i, (method, reason) in enumerate(methods)
                ]
                return {
                    "imbalance_ratio": ratio,
                    "severity": severity,
                    "suggestions": suggestions,
                }

