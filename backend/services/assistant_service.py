"""
Assistant service - interface with LangGraph Agent.
"""

import os
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

class AssistantService:
    """
    Service responsible for AI assistant communication.
    """
    def __init__(self):

        self._assistant = None
        self._agent = None
        self._initialized = False

    def _initialize(self):
        """
        Initialize assistant lazily.
        """

        if self._initialized:
            return

        try:

            from datakit.ai_assistant import create_assistant


            knowledge_base_path = os.getenv(
                "KNOWLEDGE_BASE_PATH"
            )

            if knowledge_base_path:
                knowledge_base_path = Path(
                    knowledge_base_path
                )

            self._assistant = create_assistant(
                model_name=os.getenv(
                    "OLLAMA_MODEL",
                    "mistral"
                ),
                knowledge_base_path=knowledge_base_path
            )

            self._agent = self._assistant.get(
                "agent"
            )

            self._initialized = True


            logger.info(
                "AI assistant initialized"
            )

        except Exception:

            logger.exception(
                "Assistant initialization failed"
            )

            self._initialized = False

    def chat(
        self,
        message: str
    ) -> dict:
        """
        Send message to AI assistant.
        """
        self._initialize()


        if self._agent is None:

            return {
                "success": False,
                "answer": (
                    "Assistant unavailable"
                ),
                "documents": [],
                "selected_files": []
            }

        try:

            response = self._agent.ask(
                message
            )

            return {

                "success": True,

                "answer": response.get(
                    "answer",
                    ""
                ),

                "documents": response.get(
                    "documents",
                    []
                ),

                "selected_files": response.get(
                    "selected_files",
                    []
                )

            }

        except Exception as e:

            logger.exception(
                "Assistant error"
            )

            return {

                "success": False,

                "answer": str(e),

                "documents": [],

                "selected_files": []

            }

    def is_available(self) -> bool:
        """
        Check assistant availability.
        """

        self._initialize()

        return self._agent is not None