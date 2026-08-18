"""
Semantic retrieval from Qdrant.
"""

from typing import Optional

import numpy as np

from ..embeddings.embeddings import EmbeddingModel
from .qdrant_client import QdrantVectorStore


class QdrantRetriever:
    """
    Retrieve relevant chunks using semantic/vector similarity.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: QdrantVectorStore,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> None:

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        embedding: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Perform semantic search in Qdrant.

        `top_k` overrides the retriever's default limit for a single
        call. This is used by HybridRetriever to apply its own
        `semantic_top_k` without needing a second retriever instance.
        """

        if embedding is None:
            embedding = self.embedding_model.encode_query(query)

        search_kwargs = {
            "collection_name": self.vector_store.collection_name,
            "query_vector": embedding.tolist(),
            "limit": top_k if top_k is not None else self.top_k,
        }

        if self.score_threshold is not None:
            search_kwargs["score_threshold"] = self.score_threshold

        results = self.vector_store.client.search(**search_kwargs)

        documents = []

        for rank, result in enumerate(results, start=1):

            payload = result.payload or {}

            documents.append(
                {
                    "chunk_id": payload.get("chunk_id", ""),
                    "document_id": payload.get("document_id", ""),
                    "content": payload.get("content", ""),
                    "document_name": payload.get("document_name", ""),
                    "source": payload.get("source", ""),
                    "score": float(result.score),
                    "rank": rank,
                    "retrieval_method": "semantic",
                }
            )

        return documents