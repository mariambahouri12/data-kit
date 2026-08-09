"""
Feature extraction for query classification.
"""

import numpy as np


def embedding_features(embedding: np.ndarray) -> np.ndarray:
    """
    Convert an embedding into the 2D format expected by scikit-learn.
    """
    vector = np.asarray(embedding, dtype=np.float32)

    if vector.ndim != 1:
        raise ValueError("Expected a one-dimensional embedding.")

    return vector.reshape(1, -1)