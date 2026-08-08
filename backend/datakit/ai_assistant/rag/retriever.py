"""
Retriever module.

Responsible for finding relevant knowledge
documents for a user query.

Supports:
- Global retrieval
- Filtered retrieval using Document Router
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieve relevant documents from vector store
    based on semantic similarity.
    """

    def __init__(self, embedding_model, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self._is_initialized = False

    @property
    def is_ready(self) -> bool:
        """Indique si le retriever peut servir des requêtes."""
        return self._is_initialized

    def mark_ready(self) -> None:
        """
        Marque le retriever comme prêt sans reconstruire l'index.

        FIX (#3, encapsulation) : utilisé quand le vector store a été
        chargé depuis le disque (cas où l'index existe déjà) plutôt que
        reconstruit via initialize(). Remplace l'ancien
        `retriever._is_initialized = True` fait depuis __init__.py, qui
        contournait l'encapsulation de cet attribut privé.
        """
        self._is_initialized = True

    def retrieve(
        self,
        query: str,
        selected_files: Optional[List[str]] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents.

        Args:
            query: User question
            selected_files: Files selected by DocumentRouter.
                Example: ["metrics.md", "validation.md"]
            top_k: Number of documents to retrieve

        Returns:
            List of relevant documents with metadata
        """
        if not self._is_initialized:
            return []

        try:
            query_embedding = self.embedding_model.encode([query])
            query_embedding = np.array(query_embedding).astype("float32")

            documents = self.vector_store.search(
                query_embedding,
                top_k,
                selected_files
            )
            return documents

        except Exception:
            logger.exception("Retrieval failed")
            return []

    def initialize(self, documents: List[Dict[str, Any]]) -> None:
        """
        Initialize retriever with documents (builds the index from scratch).

        Args:
            documents: List of document dictionaries
                with 'content', 'source', 'category' fields
        """
        if not documents:
            return

        try:
            texts = [doc.get("content", "") for doc in documents]
            embeddings = self.embedding_model.encode(texts)

            self.vector_store.create(embeddings, documents)
            self.vector_store.save()

            self._is_initialized = True

        except Exception:
            logger.exception("Failed to initialize retriever")
            self._is_initialized = False