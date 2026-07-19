#tabular/balance_analyser.py
from typing import Dict, Any

import pandas as pd

from ..config import BalancingMethod


class ImbalanceAnalyzer:
    """
    Analyse la distribution des classes d'une target et suggère une méthode
    de rééquilibrage adaptée. Outil de diagnostic pur : aucune modification
    de données, ne nécessite pas de ClassBalancer instancié.
    """

    # (seuil exclusif du ratio, sévérité, [(méthode, raison), ...] par priorité)
    _RULES = (
        (2, "low", [(BalancingMethod.NONE, "Les classes sont déjà équilibrées")]),
        (5, "medium", [
            (BalancingMethod.RANDOM_OVER, "Déséquilibre modéré, over-sampling simple"),
            (BalancingMethod.SMOTE, "Déséquilibre modéré, SMOTE donne de meilleurs résultats"),
        ]),
        (10, "high", [
            (BalancingMethod.SMOTE, "Déséquilibre important, SMOTE est recommandé"),
            (BalancingMethod.ADASYN, "Alternative à SMOTE pour les cas difficiles"),
        ]),
        (float("inf"), "high", [
            (BalancingMethod.ADASYN, "Déséquilibre sévère, ADASYN est recommandé"),
            (BalancingMethod.SMOTE, "Alternative pour déséquilibre sévère"),
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
                return {"imbalance_ratio": ratio, "severity": severity, "suggestions": suggestions}