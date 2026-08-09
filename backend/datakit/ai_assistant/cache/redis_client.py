"""
Redis connection management.
"""

import redis


class RedisClient:
    """Create and expose a Redis client."""

    def __init__(
        self,
        url: str = "redis://localhost:6379",
    ) -> None:
        self.client = redis.Redis.from_url(
            url,
            decode_responses=False,
        )

    def ping(self) -> bool:
        """Check Redis connectivity."""
        return bool(self.client.ping())