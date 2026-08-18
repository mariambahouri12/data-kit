"""
Knowledge-base indexing into Qdrant and Elasticsearch.
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

from .lexical.bm25_indexer import BM25Indexer


class QdrantIndexer:
    """
    Build the hybrid knowledge-base index.

    Each chunk is indexed into:
    - Qdrant       -> semantic search
    - Elasticsearch -> lexical BM25 search
    """

    def __init__(
        self,
        loader: DocumentLoader,
        processor: DocumentProcessor,
        chunker: TextChunker,
        embedding_model: EmbeddingModel,
        vector_store: QdrantVectorStore,
        bm25_indexer: BM25Indexer,
    ) -> None:

        self.loader = loader
        self.processor = processor
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.bm25_indexer = bm25_indexer

    def index(self) -> int:
        """
        Process and index all knowledge-base documents.

        The same chunks are indexed into both:
        Qdrant and Elasticsearch.
        """

        # --------------------------------------------------
        # 1. Create Qdrant collection
        # --------------------------------------------------

        self.vector_store.create_collection(
            self.embedding_model.dimension
        )

        # --------------------------------------------------
        # 2. Load documents
        # --------------------------------------------------

        documents = self.loader.load()

        if not documents:
            return 0

        document_ids = [
            document.document_id
            for document in documents
        ]

        # --------------------------------------------------
        # 3. Delete previous versions
        # --------------------------------------------------

        self._purge_existing_chunks(
            document_ids=document_ids
        )

        self.bm25_indexer.delete_documents(
            document_ids=document_ids
        )

        # --------------------------------------------------
        # 4. Process and chunk
        # --------------------------------------------------

        all_chunks = []

        for document in documents:

            processed = self.processor.process(
                document
            )

            chunks = self.chunker.split(
                processed
            )

            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        # --------------------------------------------------
        # 5. Semantic embeddings
        # --------------------------------------------------

        embeddings = self.embedding_model.encode_documents(
            [
                chunk.content
                for chunk in all_chunks
            ]
        )

        # --------------------------------------------------
        # 6. Build Qdrant points
        # --------------------------------------------------

        points = []

        for chunk, embedding in zip(
            all_chunks,
            embeddings,
        ):

            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    chunk.chunk_id,
                )
            )

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

        # --------------------------------------------------
        # 7. Insert into Qdrant
        # --------------------------------------------------

        self.vector_store.client.upsert(
            collection_name=self.vector_store.collection_name,
            points=points,
        )

        # --------------------------------------------------
        # 8. Insert same chunks into Elasticsearch
        # --------------------------------------------------

        self.bm25_indexer.index_chunks(
            all_chunks
        )

        return len(points)

    def _purge_existing_chunks(
        self,
        document_ids: list[str],
    ) -> None:
        """
        Delete every Qdrant point whose document_id
        belongs to the documents being re-indexed.
        """

        if not document_ids:
            return

        self.vector_store.client.delete(
            collection_name=self.vector_store.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchAny(
                            any=document_ids
                        ),
                    )
                ]
            ),
        )