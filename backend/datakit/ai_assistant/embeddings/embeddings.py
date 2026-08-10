"""
Embedding generation module.

Provides a single embedding model shared by:
- query classification
- Redis semantic cache
- Qdrant retrieval
- document indexing
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper around a Sentence Transformer model."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.model.get_embedding_dimension()

    def encode_query(self, text: str) -> np.ndarray:
        """Generate a normalized embedding for one query."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return np.asarray(embedding, dtype=np.float32)

    def encode_documents(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """Generate normalized embeddings for documents."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return np.asarray(embeddings, dtype=np.float32)