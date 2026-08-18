"""
Assistant service for the DataKit AI assistant.

Flow:
    User question
        ->
    AssistantOrchestrator.ask()
        ->
    standardized dict for the API
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from datakit.ai_assistant.models import AIResponse
from datakit.ai_assistant.context.context_manager import ContextManager
from datakit.ai_assistant.factory import create_assistant

from .ai_context_state import context_manager as shared_context_manager

logger = logging.getLogger(__name__)


class AssistantService:
    """
    Application-level service for the DataKit AI assistant.

    Responsibilities:
    - initialize the orchestrator lazily
    - expose chat()
    - expose dataset/preprocessing context management
    - expose assistant status
    - delegate knowledge-base indexing
    """

    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
    ) -> None:
        # Defaults to the shared ai_context_state singleton so that
        # datasets uploaded/preprocessed before any chat happens
        # (via UploadService / PreprocessingService) are visible to
        # the assistant as soon as create_assistant() runs. Passing
        # an explicit context_manager (e.g. in tests) overrides this.
        self._context_manager = context_manager or shared_context_manager

        self._orchestrator = None
        self._initialized = False
        self._init_attempted = False

    # =============================================================
    # INITIALIZATION
    # =============================================================

    def _initialize(self) -> None:
        """Lazily initialize the orchestrator."""

        if self._init_attempted:
            return

        self._init_attempted = True

        try:
            self._orchestrator = create_assistant(
                context_manager=self._context_manager,
            )

            self._initialized = self._orchestrator is not None

            if self._initialized:
                logger.info(
                    "DataKit AI orchestrator initialized successfully."
                )

        except Exception:
            self._orchestrator = None
            self._initialized = False

            logger.exception(
                "Failed to initialize DataKit AI orchestrator."
            )

    # =============================================================
    # RESET
    # =============================================================

    def reset(self) -> None:
        """Reset the orchestrator. Next call recreates it."""

        self._orchestrator = None
        self._initialized = False
        self._init_attempted = False

        logger.info("DataKit AI orchestrator reset.")

    # =============================================================
    # CHAT
    # =============================================================

    def chat(self, message: str) -> Dict[str, Any]:
        """Process a user question and return an API-ready dict."""

        message = message.strip()

        if not message:
            return self._error_response(
                question=message,
                error="Question cannot be empty.",
            )

        self._initialize()

        if self._orchestrator is None:
            return self._error_response(
                question=message,
                error=(
                    "Assistant unavailable. "
                    "Check Redis, Qdrant, classifier "
                    "and Ollama configuration."
                ),
            )

        try:
            result = self._orchestrator.ask(message)

            return self._build_response(
                question=message,
                result=result,
            )

        except Exception as exc:
            logger.exception("Chat processing failed.")

            return self._error_response(
                question=message,
                error=str(exc),
            )

    # =============================================================
    # RESPONSE
    # =============================================================

    @staticmethod
    def _build_response(
        question: str,
        result: Any,
    ) -> Dict[str, Any]:
        """
        Convert the orchestrator's AssistantResponse into the
        public API format.
        """

        response = AIResponse(
            question=question,
            answer=result.answer,
            success=True,
        )

        return {
            **response.to_dict(),
            "source": result.source,
            "similarity": result.similarity,
            "cache_hit": result.source.startswith("cache_"),
        }

    @staticmethod
    def _error_response(
        question: str,
        error: str,
    ) -> Dict[str, Any]:
        """Build a consistent error response."""

        response = AIResponse(
            question=question,
            answer="Unable to process the request.",
            success=False,
        )

        return {
            **response.to_dict(),
            "source": None,
            "similarity": None,
            "cache_hit": False,
            "error": error,
        }

    # =============================================================
    # STATUS
    # =============================================================

    def is_available(self) -> bool:
        """Return True if the orchestrator is available."""

        self._initialize()

        return self._orchestrator is not None

    def get_status(self) -> Dict[str, Any]:
        """Return basic assistant status."""

        self._initialize()

        return {
            "available": self._orchestrator is not None,
            "initialized": self._initialized,
            "init_attempted": self._init_attempted,
        }

    # =============================================================
    # CONTEXT MANAGEMENT
    # =============================================================

    def get_context(self) -> Dict[str, Any]:
        """Return the current dataset and preprocessing context."""

        return self._get_context_manager().get_full_context()

    def get_context_markdown(self) -> str:
        """Return the current context as Markdown."""

        return self._get_context_manager().to_markdown_context()

    def update_dataset(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str = "dataset",
    ) -> None:
        """Update the current dataset context."""

        self._get_context_manager().update_dataset(
            dataframe=dataframe,
            dataset_name=dataset_name,
        )

    def update_preprocessing(
        self,
        operation_name: str,
        columns: list[str],
        parameters: Optional[dict] = None,
    ) -> None:
        """Register a preprocessing operation."""

        self._get_context_manager().update_preprocessing(
            operation_name=operation_name,
            columns=columns,
            parameters=parameters,
        )

    def _get_context_manager(self) -> ContextManager:
        """
        Return the configured context manager.

        Since __init__ always falls back to the shared
        ai_context_state singleton, self._context_manager is never
        None here — but the orchestrator's own context_manager is
        preferred once initialized, in case create_assistant() ever
        substitutes a different instance internally.
        """

        if self._orchestrator is not None:
            context_manager = getattr(
                self._orchestrator,
                "context_manager",
                None,
            )

            if context_manager is not None:
                return context_manager

        return self._context_manager

    # =============================================================
    # KNOWLEDGE BASE
    # =============================================================

    def rebuild_index(self) -> bool:
        """Rebuild the Qdrant knowledge-base index."""

        self._initialize()

        if self._orchestrator is None:
            logger.error("Cannot rebuild index: assistant unavailable.")
            return False

        indexer = getattr(
            self._orchestrator,
            "indexer",
            None,
        )

        if indexer is None:
            logger.error("Qdrant indexer is not configured.")
            return False

        try:
            indexed_count = indexer.index()

            logger.info(
                "Qdrant index rebuilt: %d chunks.",
                indexed_count,
            )

            return True

        except Exception:
            logger.exception("Qdrant index rebuild failed.")
            return False


# =============================================================
# SINGLETON
# =============================================================

assistant_service = AssistantService()


__all__ = [
    "AssistantService",
    "assistant_service",
]