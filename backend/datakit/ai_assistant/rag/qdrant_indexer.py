"""
Knowledge-base indexing into Qdrant.
"""

from uuid import uuid5, NAMESPACE_URL

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
)

from .chunker import TextChunker
from .document_loader import DocumentLoader
from .document_processor import DocumentProcessor
from .qdrant_client import QdrantVectorStore
from ..embeddings.embeddings import EmbeddingModel


class QdrantIndexer:
    """Build the Qdrant knowledge-base index."""

    def __init__(
        self,
        loader: DocumentLoader,
        processor: DocumentProcessor,
        chunker: TextChunker,
        embedding_model: EmbeddingModel,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.loader = loader
        self.processor = processor
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def index(self) -> int:
        """
        Process and index all knowledge-base documents.

        """

        self.vector_store.create_collection(
            self.embedding_model.dimension
        )

        documents = self.loader.load()

        if not documents:
            return 0

        self._purge_existing_chunks(
            document_ids=[doc.document_id for doc in documents]
        )

        all_chunks = []

        for document in documents:
            processed = self.processor.process(document)
            all_chunks.extend(self.chunker.split(processed))

        if not all_chunks:
            return 0

        embeddings = self.embedding_model.encode_documents(
            [chunk.content for chunk in all_chunks]
        )

        points = []

        for chunk, embedding in zip(all_chunks, embeddings):
            point_id = str(uuid5(NAMESPACE_URL, chunk.chunk_id))

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "document_name": chunk.document_name,
                        "content": chunk.content,
                        "source": chunk.source,
                    },
                )
            )

        self.vector_store.client.upsert(
            collection_name=self.vector_store.collection_name,
            points=points,
        )

        return len(points)

    def _purge_existing_chunks(
        self,
        document_ids: list[str],
    ) -> None:
        """
        Delete every point whose `document_id` payload matches one
        of the documents about to be re-indexed.
        """

        if not document_ids:
            return

        self.vector_store.client.delete(
            collection_name=self.vector_store.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchAny(any=document_ids),
                    )
                ]
            ),
        )