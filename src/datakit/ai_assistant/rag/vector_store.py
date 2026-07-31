# datakit/ai_assistant/rag/vector_store.py

"""
FAISS vector storage module.
"""

import os
import pickle
from typing import List, Dict, Any

import faiss
import numpy as np


class VectorStore:
    """
    FAISS-based vector storage for document retrieval.
    """

    def __init__(
        self,
        index_path: str = "storage/rag_index/index.faiss",
        metadata_path: str = "storage/rag_index/metadata.pkl",
        dimension: int = 384  # all-MiniLM-L6-v2 dimension
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = dimension
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

    def create(
        self,
        embeddings: np.ndarray,
        documents: List[str]
    ) -> None:
        """Create FAISS index from embeddings."""
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype("float32"))
        
        # Store document metadata
        self.metadata = [
            {"content": doc, "id": i}
            for i, doc in enumerate(documents)
        ]

    def save(self) -> None:
        """Save index and metadata to disk."""
        os.makedirs(
            os.path.dirname(self.index_path),
            exist_ok=True
        )

        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, "wb") as file:
            pickle.dump(self.metadata, file)

    def load(self) -> bool:
        """Load index and metadata from disk."""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                self.dimension = self.index.d

                with open(self.metadata_path, "rb") as file:
                    self.metadata = pickle.load(file)
                return True
        except Exception as e:
            print(f"Failed to load vector store: {e}")

        return False

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve nearest documents."""
        if self.index is None:
            return []

        scores, indices = self.index.search(
            query_embedding.astype("float32"),
            top_k
        )

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(scores[0][i])
                results.append(result)

        return results

    def is_ready(self) -> bool:
        """Check if vector store is initialized."""
        return self.index is not None and len(self.metadata) > 0