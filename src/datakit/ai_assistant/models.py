# datakit/ai_assistant/models.py

"""
Data models for AI Assistant.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class PreprocessingOperation:
    """Represents a preprocessing operation."""
    operation: str
    columns: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "columns": self.columns,
            "parameters": self.parameters,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingOperation":
        return cls(
            operation=data.get("operation", ""),
            columns=data.get("columns", []),
            parameters=data.get("parameters", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )


@dataclass
class AIResponse:
    """Represents an AI assistant response."""
    question: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    confidence: Optional[float] = None
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "success": self.success,
            "confidence": self.confidence,
            "suggestions": self.suggestions
        }


@dataclass
class DatasetInfo:
    """Represents dataset information."""
    name: str
    rows: int
    columns: int
    column_info: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": {"rows": self.rows, "columns": self.columns},
            "columns": self.column_info,
            "quality": self.quality_metrics
        }


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}