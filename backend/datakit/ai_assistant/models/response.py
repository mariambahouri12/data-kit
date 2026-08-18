"""
Response model returned by the assistant orchestrator.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AssistantResponse:
    answer: str
    source: str
    similarity: Optional[float] = None