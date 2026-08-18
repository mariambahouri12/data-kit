"""
Elasticsearch client used for lexical BM25 retrieval.
"""

from elasticsearch import Elasticsearch


class BM25Client:
    """
    Manage the Elasticsearch index used for lexical retrieval.

    Elasticsearch automatically maintains an inverted index
    for the text fields.
    """

    def __init__(
        self,
        url: str = "http://localhost:9200",
        index_name: str = "datakit_knowledge_bm25",
    ) -> None:

        self.index_name = index_name

        self.client = Elasticsearch(
            url
        )

    def create_index(self) -> None:
        """
        Create the BM25 index if it does not already exist.

        The `content` field is analyzed and indexed by
        Elasticsearch's inverted index.
        """

        if self.client.indices.exists(
            index=self.index_name
        ):
            return

        mappings = {
            "properties": {
                "chunk_id": {
                    "type": "keyword"
                },
                "document_id": {
                    "type": "keyword"
                },
                "document_name": {
                    "type": "keyword"
                },
                "content": {
                    "type": "text"
                },
                "source": {
                    "type": "keyword"
                },
            }
        }

        self.client.indices.create(
            index=self.index_name,
            mappings=mappings,
        )

    def delete_documents(
        self,
        document_ids: list[str],
    ) -> None:
        """
        Delete all chunks belonging to the given documents.
        """

        if not document_ids:
            return

        if not self.client.indices.exists(
            index=self.index_name
        ):
            return

        self.client.delete_by_query(
            index=self.index_name,
            query={
                "terms": {
                    "document_id": document_ids
                }
            },
            refresh=True,
        )

    def index_document(
        self,
        document: dict,
    ) -> None:
        """Index one chunk."""

        self.client.index(
            index=self.index_name,
            id=document["chunk_id"],
            document=document,
        )

    def refresh(self) -> None:
        """Make recently indexed documents searchable."""

        self.client.indices.refresh(
            index=self.index_name
        )