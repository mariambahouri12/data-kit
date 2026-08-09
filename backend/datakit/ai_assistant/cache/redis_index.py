"""
Redis vector index management.
"""

from redis import Redis
from redis.exceptions import ResponseError


class RedisVectorIndex:
    """Manage Redis vector-search indexes."""

    def __init__(
        self,
        client: Redis,
        dimension: int,
        distance_metric: str = "COSINE",
    ) -> None:
        self.client = client
        self.dimension = dimension
        self.distance_metric = distance_metric

    def create_indexes(self) -> None:
        """Create shared and private vector indexes."""
        self._create_index(
            index_name="idx:cache:shared",
            key_prefix="cache:shared:",
        )

        self._create_index(
            index_name="idx:cache:private",
            key_prefix="cache:private:",
        )

    def _create_index(
        self,
        index_name: str,
        key_prefix: str,
    ) -> None:

        schema = [
            "ON",
            "HASH",
            "PREFIX",
            "1",
            key_prefix,
            "SCHEMA",
            "question",
            "TEXT",
            "answer",
            "TEXT",
            "dataset_fingerprint",
            "TAG",
            "scope",
            "TAG",
            "embedding",
            "VECTOR",
            "HNSW",
            "6",
            "TYPE",
            "FLOAT32",
            "DIM",
            str(self.dimension),
            "DISTANCE_METRIC",
            self.distance_metric,
        ]

        try:
            self.client.execute_command(
                "FT.CREATE",
                index_name,
                *schema,
            )

        except ResponseError as exc:
            if "Index already exists" not in str(exc):
                raise