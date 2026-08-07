"""
Retriever module.

Responsible for finding relevant knowledge
documents for a user query.

Supports:
- Global retrieval
- Filtered retrieval using Document Router
"""

import numpy as np
from typing import List, Dict, Any, Optional

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
        selected_files: Optional[List[str]] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents.

        Args:
            query:
                User question

            selected_files:
                Files selected by DocumentRouter.
                Example:
                [
                    "metrics.md",
                    "validation.md"
                ]

            top_k:
                Number of documents to retrieve

        Returns:
            List of relevant documents with metadata
        """

        if not self._is_initialized:
            # Return empty list instead of fake content
            return []

        try:
            # Create query embedding
            query_embedding = (
                self.embedding_model
                .encode([query])
            )

            query_embedding = np.array(
                query_embedding
            ).astype("float32")

            # Search with optional file filtering
            documents = (
                self.vector_store.search(
                    query_embedding,
                    top_k,
                    selected_files
                )
            )

            return documents

        except Exception as e:
            # Return empty list on error
            return []

    def _fallback_retrieval(
        self,
        query: str,
        error: str = None
    ) -> List[Dict[str, Any]]:
        """
        Fallback when vector store
        is not available.
        """

        # Return empty list - prompt_manager will handle it
        return []

    def initialize(
        self,
        documents: List[Dict[str, Any]]
    ) -> None:
        """
        Initialize retriever with documents.

        Args:
            documents: List of document dictionaries
            with 'content', 'source', 'category' fields
        """

        if not documents:
            return

        try:
            # Extract text content for embeddings
            texts = [
                doc.get("content", "")
                for doc in documents
            ]

            # Generate embeddings
            embeddings = (
                self.embedding_model
                .encode(texts)
            )

            # Create vector index with metadata
            self.vector_store.create(
                embeddings,
                documents
            )

            # Save index
            self.vector_store.save()

            self._is_initialized = True

        except Exception as e:
            print(
                f"Failed to initialize retriever: {e}"
            )
            self._is_initialized = False