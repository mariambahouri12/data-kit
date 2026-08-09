from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheEntry:
    question: str
    answer: str
    scope: str
    dataset_fingerprint: Optional[str] = None
    similarity: Optional[float] = None