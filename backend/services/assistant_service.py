"""
Assistant service - interface avec LangGraph Agent.
"""

from pathlib import Path
import os
from typing import Optional


class AssistantService:
    """Service pour l'assistant IA avec LangGraph."""

    def __init__(self):
        self._assistant = None
        self._agent = None
        self._initialized = False

    def _initialize(self):
        """Initialiser l'assistant au premier appel (lazy loading)."""
        if self._initialized:
            return
        
        try:
            from datakit.ai_assistant import create_assistant
            
            # Trouver le chemin de la knowledge base
            knowledge_base_path = None
            possible_paths = [
                Path("knowledge_base"),
                Path("../knowledge_base"),
                Path(os.getenv("KNOWLEDGE_BASE_PATH", "")),
                Path(__file__).parent.parent / "knowledge_base",
            ]
            
            for path in possible_paths:
                if path and path.exists():
                    knowledge_base_path = str(path)
                    break

            # Créer l'assistant
            self._assistant = create_assistant(
                model_name=os.getenv("OLLAMA_MODEL", "mistral"),
                knowledge_base_path=knowledge_base_path
            )
            self._agent = self._assistant.get("agent")
            self._initialized = True

            print(f"✅ Assistant initialisé (KB: {knowledge_base_path})")

        except Exception as e:
            print(f"❌ Erreur d'initialisation: {e}")
            self._initialized = False

    def chat(self, message: str) -> dict:
        """
        Envoyer un message à l'assistant.

        Args:
            message: Message de l'utilisateur

        Returns:
            Réponse de l'assistant
        """
        # Initialiser au premier appel
        self._initialize()
        
        if not self._initialized or self._agent is None:
            return {
                "answer": "⚠️ L'assistant n'est pas disponible. Vérifiez qu'Ollama est installé et que le modèle 'mistral' est téléchargé.",
                "documents": [],
                "selected_files": [],
                "success": False
            }

        try:
            response = self._agent.ask(message)
            return {
                "answer": response.get("answer", "Désolé, je n'ai pas pu générer une réponse."),
                "documents": response.get("documents", []),
                "selected_files": response.get("selected_files", []),
                "success": response.get("success", False)
            }
        except Exception as e:
            return {
                "answer": f"❌ Erreur: {str(e)}",
                "documents": [],
                "selected_files": [],
                "success": False
            }

    def is_available(self) -> bool:
        """Vérifier si l'assistant est disponible."""
        return self._initialized and self._agent is not None