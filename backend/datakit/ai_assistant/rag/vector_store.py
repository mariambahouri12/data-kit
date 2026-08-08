"""
FAISS vector storage module.

Supports:
- semantic similarity search
- metadata filtering for Agentic RAG
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

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
        dimension: int = 384,
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = dimension

        self.index = None
        self.metadata: List[Dict[str, Any]] = []

    @property
    def exists(self) -> bool:
        """
        Check whether the FAISS index exists on disk.
        """

        return os.path.exists(self.index_path)

    def create(
        self,
        embeddings: np.ndarray,
        documents: List[Any],
    ) -> None:
        """
        Create a FAISS index.

        Documents may be dictionaries or strings.
        Existing metadata fields are preserved.
        """

        if embeddings is None or len(embeddings) == 0:
            raise ValueError(
                "Cannot create FAISS index from empty embeddings."
            )

        self.dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.index.add(
            embeddings.astype("float32")
        )

        self.metadata = []

        for i, doc in enumerate(documents):

            if isinstance(doc, dict):

                metadata = doc.copy()

                metadata.setdefault(
                    "content",
                    "",
                )

                metadata.setdefault(
                    "source",
                    "unknown",
                )

                metadata.setdefault(
                    "category",
                    "unknown",
                )

                metadata.setdefault(
                    "id",
                    i,
                )

            else:

                metadata = {
                    "id": i,
                    "content": str(doc),
                    "source": "unknown",
                    "category": "unknown",
                }

            self.metadata.append(metadata)

    def save(self) -> None:
        """
        Persist FAISS index and metadata.
        """

        directory = os.path.dirname(
            self.index_path
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        metadata_directory = os.path.dirname(
            self.metadata_path
        )

        if metadata_directory:
            os.makedirs(
                metadata_directory,
                exist_ok=True,
            )

        if self.index is not None:
            faiss.write_index(
                self.index,
                self.index_path,
            )

        with open(
            self.metadata_path,
            "wb",
        ) as file:
            pickle.dump(
                self.metadata,
                file,
            )

    def load(self) -> bool:
        """
        Load FAISS index and metadata from disk.
        """

        try:

            if not self.exists:
                return False

            if not os.path.exists(
                self.metadata_path
            ):
                logger.warning(
                    "FAISS index exists but metadata file is missing."
                )
                return False

            self.index = faiss.read_index(
                self.index_path
            )

            self.dimension = self.index.d

            with open(
                self.metadata_path,
                "rb",
            ) as file:
                self.metadata = pickle.load(file)

            if not isinstance(
                self.metadata,
                list,
            ):
                logger.warning(
                    "Invalid metadata format."
                )
                self.index = None
                self.metadata = []
                return False

            return True

        except Exception:
            logger.exception(
                "Failed to load vector store."
            )

            self.index = None
            self.metadata = []

            return False

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        selected_files: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve nearest documents.

        When selected_files is provided, FAISS retrieves
        a larger candidate set and the results are then
        filtered using document metadata.
        """

        if self.index is None:
            return []

        if not self.metadata:
            return []

        if query_embedding is None:
            return []

        if top_k <= 0:
            return []

        # Retrieve extra candidates when filtering is active.
        if selected_files:
            search_k = min(
                len(self.metadata),
                max(top_k * 5, top_k),
            )
        else:
            search_k = min(
                len(self.metadata),
                top_k,
            )

        if search_k <= 0:
            return []

        scores, indices = self.index.search(
            query_embedding.astype("float32"),
            search_k,
        )

        results = []

        selected_set = set(
            selected_files or []
        )

        for i, idx in enumerate(indices[0]):

            if idx < 0:
                continue

            if idx >= len(self.metadata):
                continue

            document = self.metadata[idx].copy()

            # -------------------------------------------------
            # Metadata filtering
            # -------------------------------------------------

            if selected_set:

                source = document.get(
                    "source",
                    "",
                )

                if source not in selected_set:
                    continue

            document["score"] = float(
                scores[0][i]
            )

            results.append(document)

            if len(results) >= top_k:
                break

        return results

    def is_ready(self) -> bool:
        """
        Check whether the vector store is initialized.
        """

        return (
            self.index is not None
            and len(self.metadata) > 0
        )