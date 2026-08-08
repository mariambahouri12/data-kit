"""
Retriever module.

Responsible for finding relevant knowledge documents.

Supports:
- global retrieval
- filtered retrieval using Document Router
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieve relevant documents from the vector store
    based on semantic similarity.
    """

    def __init__(
        self,
        embedding_model,
        vector_store,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self._is_initialized = False

    @property
    def is_ready(self) -> bool:
        """
        Indicate whether the retriever is ready.
        """

        return self._is_initialized

    def mark_ready(self) -> None:
        """
        Mark the retriever as ready after loading
        an existing vector store.
        """

        self._is_initialized = True

    def retrieve(
        self,
        query: str,
        selected_files: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents.
        """

        if not self._is_initialized:
            return []

        if not query or not query.strip():
            return []

        try:

            query_embedding = self.embedding_model.encode(
                [query]
            )

            query_embedding = np.asarray(
                query_embedding,
                dtype="float32",
            )

            documents = self.vector_store.search(
                query_embedding,
                top_k,
                selected_files,
            )

            return documents

        except Exception:
            logger.exception(
                "Retrieval failed."
            )
            return []

    def initialize(
        self,
        documents: List[Dict[str, Any]],
    ) -> None:
        """
        Build and persist the vector index.
        """

        if not documents:
            self._is_initialized = False
            return

        try:

            texts = [
                doc.get("content", "")
                for doc in documents
            ]

            embeddings = self.embedding_model.encode(
                texts
            )

            self.vector_store.create(
                embeddings,
                documents,
            )

            self.vector_store.save()

            self._is_initialized = True

        except Exception:
            logger.exception(
                "Failed to initialize retriever."
            )

            self._is_initialized = False