"""
FAISS vector storage module.

Supports:
- Semantic similarity search
- Metadata filtering for Agentic RAG
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector storage for document retrieval.
    """

    def __init__(
        self,
        index_path: str = "storage/rag_index/index.faiss",
        metadata_path: str = "storage/rag_index/metadata.pkl",
        dimension: int = 384
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = dimension

        self.index = None
        self.metadata: List[Dict[str, Any]] = []

    @property
    def exists(self) -> bool:
        """Check if FAISS index exists on disk."""
        return os.path.exists(self.index_path)

    def create(self, embeddings: np.ndarray, documents: List[Any]) -> None:
        """
        Create FAISS index.

        Documents can be:
        - strings
        - dictionaries with metadata
        """
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype("float32"))

        self.metadata = []

        for i, doc in enumerate(documents):
            if isinstance(doc, dict):
                metadata = {
                    "content": doc.get("content", ""),
                    "source": doc.get("source", "unknown"),
                    "category": doc.get("category", "unknown"),
                    "id": i
                }
            else:
                metadata = {
                    "content": doc,
                    "source": "unknown",
                    "category": "unknown",
                    "id": i
                }

            self.metadata.append(metadata)

    def save(self) -> None:
        """Save FAISS index and metadata."""
        directory = os.path.dirname(self.index_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, "wb") as file:
            pickle.dump(self.metadata, file)

    def load(self) -> bool:
        """Load FAISS index and metadata."""
        try:
            if self.exists:
                self.index = faiss.read_index(self.index_path)
                self.dimension = self.index.d

                with open(self.metadata_path, "rb") as file:
                    self.metadata = pickle.load(file)

                return True

        except Exception:
            # FIX (#6) : logging cohérent avec le reste du module,
            # au lieu d'un print() non capturé par la config de logging.
            logger.exception("Failed to load vector store")

        return False

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        selected_files: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve nearest documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            selected_files: Files selected by DocumentRouter.
                Example: ["metrics.md", "validation.md"]
        """
        if self.index is None:
            return []

        search_k = top_k
        if selected_files:
            search_k = min(len(self.metadata), top_k * 5)

        scores, indices = self.index.search(
            query_embedding.astype("float32"),
            search_k
        )

        results = []

        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            if idx >= len(self.metadata):
                continue

            document = self.metadata[idx].copy()

            if selected_files:
                source = document.get("source", "")
                source_matches = False
                for selected_file in selected_files:
                    if source == selected_file or source.endswith(selected_file):
                        source_matches = True
                        break

                if not source_matches:
                    continue

            document["score"] = float(scores[0][i])
            results.append(document)

            if len(results) >= top_k:
                break

        return results

    def is_ready(self) -> bool:
        """Check if vector store is initialized."""
        return self.index is not None and len(self.metadata) > 0