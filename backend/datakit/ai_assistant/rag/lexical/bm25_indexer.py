"""
BM25 lexical indexer.
"""

from elasticsearch.helpers import bulk

from .bm25_client import BM25Client


class BM25Indexer:
    """
    Index document chunks into Elasticsearch.

    Elasticsearch maintains the inverted index internally
    and uses BM25 for lexical scoring.
    """

    def __init__(
        self,
        client: BM25Client,
    ) -> None:

        self.client = client

    def index_chunks(
        self,
        chunks: list,
    ) -> int:
        """
        Index a list of DocumentChunk objects.
        """

        if not chunks:
            return 0

        self.client.create_index()

        actions = []

        for chunk in chunks:

            actions.append(
                {
                    "_index": self.client.index_name,
                    "_id": chunk.chunk_id,
                    "_source": {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "document_name": chunk.document_name,
                        "content": chunk.content,
                        "source": chunk.source,
                    },
                }
            )

        bulk(
            self.client.client,
            actions,
        )

        self.client.refresh()

        return len(actions)

    def delete_documents(
        self,
        document_ids: list[str],
    ) -> None:
        """
        Delete all indexed chunks belonging to documents.
        """

        self.client.delete_documents(
            document_ids
        )