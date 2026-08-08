"""
Assistant service - interface with LangGraph Agent.
"""

import os
import logging
from pathlib import Path

from datakit.ai_assistant.models import AIResponse
from .ai_context_state import context_manager as shared_context_manager

logger = logging.getLogger(__name__)


class AssistantService:
    """
    Service responsible for AI assistant communication.
    """

    def __init__(self):
        self._assistant = None
        self._agent = None
        self._initialized = False
        self._init_attempted = False

    def _initialize(self):
        if self._init_attempted:
            return
        self._init_attempted = True

        try:
            from datakit.ai_assistant import create_assistant

            knowledge_base_path = os.getenv("KNOWLEDGE_BASE_PATH")
            if knowledge_base_path:
                knowledge_base_path = Path(knowledge_base_path)

            self._assistant = create_assistant(
                model_name=os.getenv("OLLAMA_MODEL", "mistral"),
                knowledge_base_path=knowledge_base_path,
                context_manager=shared_context_manager
            )

            self._agent = self._assistant.get("agent")
            self._initialized = True

            logger.info("AI assistant initialized")

        except Exception:
            logger.exception("Assistant initialization failed")
            self._initialized = False

    def reset(self) -> None:
        self._assistant = None
        self._agent = None
        self._initialized = False
        self._init_attempted = False

    def chat(self, message: str) -> dict:
        self._initialize()

        if self._agent is None:
            response = AIResponse(
                question=message,
                answer="Assistant unavailable",
                success=False
            )
            return {
                **response.to_dict(),
                "documents": [],
                "selected_files": []
            }

        try:
            result = self._agent.ask(message)

            response = AIResponse(
                question=message,
                answer=result.get("answer", ""),
                success=result.get("success", False)
            )
            return {
                **response.to_dict(),
                "documents": result.get("documents", []),
                "selected_files": result.get("selected_files", [])
            }

        except Exception as e:
            logger.exception("Assistant error")
            response = AIResponse(
                question=message,
                answer=str(e),
                success=False
            )
            return {
                **response.to_dict(),
                "documents": [],
                "selected_files": []
            }

    def is_available(self) -> bool:
        self._initialize()
        return self._agent is not None