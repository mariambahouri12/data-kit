"""
Machine-learning classifier used to select the cache scope.

Possible scopes:
- private
- shared
"""

from pathlib import Path

import joblib
import numpy as np

from .features import embedding_features


class QueryClassifier:
    """Classify a query into private or shared scope."""

    PRIVATE = "private"
    SHARED = "shared"

    def __init__(self, model_path: str) -> None:
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Classifier model not found: {path}"
            )

        self.model = joblib.load(path)

    def predict(self, embedding: np.ndarray) -> str:
        """Predict the cache scope."""
        features = embedding_features(embedding)
        prediction = self.model.predict(features)[0]

        if prediction not in {
            self.PRIVATE,
            self.SHARED,
        }:
            raise ValueError(
                f"Invalid classifier output: {prediction}"
            )

        return prediction

    def predict_proba(
        self,
        embedding: np.ndarray,
    ) -> dict[str, float]:
        """Return class probabilities when supported."""
        if not hasattr(self.model, "predict_proba"):
            return {}

        features = embedding_features(embedding)
        probabilities = self.model.predict_proba(features)[0]

        return {
            label: float(probability)
            for label, probability in zip(
                self.model.classes_,
                probabilities,
            )
        }