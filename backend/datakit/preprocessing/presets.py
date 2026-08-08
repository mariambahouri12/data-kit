
# preprocessing/presets.py

from typing import Dict, Any, List


class PreprocessingPresets:
    """Configuration presets for different use cases."""

    @staticmethod
    def get_preset(name: str) -> Dict[str, Any]:
        """
        Get a configuration preset.

        Args:
            name: Name of the preset.

        Returns:
            Configuration dictionary.
        """
        presets = {
            "quick": {
                "imputation_method": "median",
                "scaling_method": "standard",
                "encoding_method": "onehot",
                "outlier_method": "iqr",
                "outlier_threshold": 1.5,
            },
            "robust": {
                "imputation_method": "median",
                "scaling_method": "robust",
                "encoding_method": "target",
                "outlier_method": "iqr",
                "outlier_threshold": 3.0,
                "outlier_action": "winsorize",
            },
            "high_performance": {
                "imputation_method": "knn",
                "scaling_method": "standard",
                "encoding_method": "catboost",
                "outlier_method": "isolation_forest",
                "outlier_action": "winsorize",
                "create_polynomial": True,
                "polynomial_degree": 2,
                "apply_boxcox": True,
            },
            "minimal": {
                "imputation_method": "median",
                "scaling_method": "standard",
                "encoding_method": "ordinal",
                "outlier_method": "none",
                "drop_duplicates": False,
                "drop_high_missing": False,
            },
            "nlp_ready": {
                "imputation_method": "most_frequent",
                "scaling_method": "standard",
                "encoding_method": "frequency",
                "outlier_method": "iqr",
                "outlier_threshold": 1.5,
                "create_polynomial": False,
            },
            "time_series": {
                "imputation_method": "median",
                "scaling_method": "standard",
                "encoding_method": "onehot",
                "outlier_method": "iqr",
                "outlier_threshold": 1.5,
                "drop_duplicates": True,
                "create_interactions": False,
                "create_ratios": False,
            },
            "imbalanced": {
                "imputation_method": "median",
                "scaling_method": "standard",
                "encoding_method": "target",
                "outlier_method": "iqr",
                "outlier_threshold": 1.5,
                "balancing_method": "smote",
                "balancing_apply_before_pipeline": True,
                "outlier_action": "winsorize",
            },
        }

        if name not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(
                f"Preset '{name}' not found. Available: {available}"
            )

        return presets[name]

    @staticmethod
    def list_presets() -> List[str]:
        """List all available configuration presets."""
        return [
            "quick",
            "robust",
            "high_performance",
            "minimal",
            "nlp_ready",
            "time_series",
            "imbalanced",
        ]

