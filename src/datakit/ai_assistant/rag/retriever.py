# datakit/ai_assistant/rag/retriever.py

"""
Retriever module.

Responsible for finding relevant knowledge
documents for a user query.
"""

import numpy as np
from typing import List, Dict, Any


class Retriever:
    """
    Retrieve relevant documents from vector store
    based on semantic similarity.
    """

    def __init__(
        self,
        embedding_model,
        vector_store
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self._is_initialized = False

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents.

        Args:
            query: User question
            top_k: Number of documents to retrieve

        Returns:
            List of relevant documents with metadata
        """
        if not self._is_initialized:
            return self._fallback_retrieval(query)

        try:
            query_embedding = self.embedding_model.encode([query])
            query_embedding = np.array(query_embedding).astype("float32")

            documents = self.vector_store.search(
                query_embedding,
                top_k
            )

            return documents

        except Exception as e:
            return self._fallback_retrieval(query, error=str(e))

    def _fallback_retrieval(
        self,
        query: str,
        error: str = None
    ) -> List[Dict[str, Any]]:
        """
        Fallback when vector store is not available.
        Returns documents with default content.
        """
        return [{
            "content": f"Knowledge base not ready. Error: {error}",
            "source": "fallback"
        }]

    def initialize(self, documents: List[str]) -> None:
        """
        Initialize retriever with documents.
        """
        if not documents:
            return

        try:
            embeddings = self.embedding_model.encode(documents)
            self.vector_store.create(embeddings, documents)
            self.vector_store.save()
            self._is_initialized = True
        except Exception as e:
            print(f"Failed to initialize retriever: {e}")
            self._is_initialized = False