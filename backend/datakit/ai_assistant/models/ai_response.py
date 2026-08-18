"""
API-facing response model for the DataKit AI assistant.

Distinct from AssistantResponse (models/response.py), which is the
internal result returned by AssistantOrchestrator.ask(). AIResponse
is the shape exposed to the API/frontend by AssistantService.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class AIResponse:
    """Standardized API response for a chat interaction."""

    question: str
    answer: str
    success: bool

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this response to a plain dict for the API."""
        return asdict(self)