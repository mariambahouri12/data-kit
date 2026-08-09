"""
Semantic retrieval from Qdrant.
"""

import numpy as np

from ..embeddings.embeddings import EmbeddingModel
from .qdrant_client import QdrantVectorStore


class QdrantRetriever:
    """Retrieve relevant knowledge-base chunks."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: QdrantVectorStore,
        top_k: int = 5,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
    ) -> list[dict]:
        """Retrieve the most relevant chunks."""

        embedding = self.embedding_model.encode_query(
            query
        )

        results = self.vector_store.client.search(
            collection_name=(
                self.vector_store.collection_name
            ),
            query_vector=embedding.tolist(),
            limit=self.top_k,
        )

        documents = []

        for result in results:
            payload = result.payload or {}

            documents.append(
                {
                    "content": payload.get(
                        "content",
                        "",
                    ),
                    "document_name": payload.get(
                        "document_name",
                        "",
                    ),
                    "source": payload.get(
                        "source",
                        "",
                    ),
                    "score": float(
                        result.score
                    ),
                }
            )

        return documents