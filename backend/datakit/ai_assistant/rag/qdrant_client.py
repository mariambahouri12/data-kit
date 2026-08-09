"""
Qdrant client management.
"""

from qdrant_client import QdrantClient


class QdrantVectorStore:
    """Manage the Qdrant collection used by DataKit."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "datakit_knowledge",
    ) -> None:
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def create_collection(
        self,
        dimension: int,
    ) -> None:
        """Create the collection if it does not exist."""

        from qdrant_client.models import Distance
        from qdrant_client.models import VectorParams

        collections = self.client.get_collections()

        exists = any(
            collection.name == self.collection_name
            for collection in collections.collections
        )

        if exists:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
            ),
        )