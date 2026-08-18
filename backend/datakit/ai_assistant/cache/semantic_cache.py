"""
Semantic cache backed by Redis vector search.
"""

import uuid
from typing import Optional

import numpy as np
from redis import Redis

from ..models.cache import CacheEntry


class SemanticCache:
    """Semantic cache using Redis vector indexes."""

    def __init__(
        self,
        client: Redis,
        dimension: int,
        similarity_threshold: float = 0.85,
    ) -> None:
        self.client = client
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold

    def search(
        self,
        embedding: np.ndarray,
        scope: str,
        dataset_fingerprint: Optional[str] = None,
    ) -> Optional[CacheEntry]:
        """
        Search for a semantically similar cached question.
        """

        if scope not in {"private", "shared"}:
            raise ValueError(f"Unsupported cache scope: {scope}")

        if scope == "private" and not dataset_fingerprint:
            return None

        index_name = f"idx:cache:{scope}"

        if scope == "private":
            query = (
                f"(@dataset_fingerprint:{{{dataset_fingerprint}}})"
                f"=>[KNN 1 @embedding $vector AS vector_distance]"
            )
        else:
            query = "(*)=>[KNN 1 @embedding $vector AS vector_distance]"

        vector_bytes = np.asarray(
            embedding,
            dtype=np.float32,
        ).tobytes()

        result = self.client.execute_command(
            "FT.SEARCH",
            index_name,
            query,
            "PARAMS",
            "2",
            "vector",
            vector_bytes,
            "DIALECT",
            "2",
            "SORTBY",
            "vector_distance",
            "RETURN",
            "5",
            "question",
            "answer",
            "scope",
            "dataset_fingerprint",
            "vector_distance",
            "LIMIT",
            "0",
            "1",
        )

        if not result or result[0] == 0:
            return None

        fields = self._parse_fields(result[2])

        distance = float(fields.get("vector_distance", 1.0))

        similarity = 1.0 - distance

        if similarity < self.similarity_threshold:
            return None

        return CacheEntry(
            question=fields["question"],
            answer=fields["answer"],
            scope=fields["scope"],
            dataset_fingerprint=fields.get("dataset_fingerprint"),
            similarity=similarity,
        )

    def store(
        self,
        question: str,
        answer: str,
        embedding: np.ndarray,
        scope: str,
        dataset_fingerprint: Optional[str] = None,
    ) -> None:
        """Store a question-answer pair in Redis."""

        if scope not in {"private", "shared"}:
            raise ValueError(f"Unsupported cache scope: {scope}")

        if scope == "private" and not dataset_fingerprint:
            raise ValueError(
                "Private cache entries require a dataset fingerprint."
            )

        key = f"cache:{scope}:{uuid.uuid4().hex}"

        self.client.hset(
            key,
            mapping={
                "question": question,
                "answer": answer,
                "scope": scope,
                "dataset_fingerprint": dataset_fingerprint or "",
                "embedding": np.asarray(
                    embedding,
                    dtype=np.float32,
                ).tobytes(),
            },
        )

    @staticmethod
    def _parse_fields(raw_fields: list) -> dict:
        """Convert a flat Redis field/value list into a dictionary."""

        fields = {}

        for index in range(0, len(raw_fields), 2):
            key = raw_fields[index]
            value = raw_fields[index + 1]

            if isinstance(key, bytes):
                key = key.decode("utf-8")

            if isinstance(value, bytes):
                value = value.decode("utf-8")

            fields[key] = value

        return fields