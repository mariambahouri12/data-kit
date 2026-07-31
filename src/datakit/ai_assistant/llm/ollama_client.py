# datakit/ai_assistant/llm/ollama_client.py

"""
Ollama client for DataKit AI Assistant.
Version robuste utilisant subprocess pour éviter les problèmes d'API.
"""

import logging
import subprocess
from typing import Dict, Any, Generator

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client to interact with local Ollama LLM using subprocess."""

    def __init__(
        self,
        model_name: str = "mistral",
        host: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.host = host
        self._available = False
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if Ollama is available using subprocess."""
        try:
            # Vérifier d'abord si ollama est installé
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if result.returncode != 0:
                logger.warning("Ollama not installed or not in PATH")
                self._available = False
                return
            
            # Lister les modèles
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                logger.warning(f"Ollama list failed: {result.stderr}")
                self._available = False
                return
            
            # Parser the output
            lines = result.stdout.strip().split("\n")
            logger.debug(f"Ollama list output: {lines}")
            
            # find the model
            available_models = []
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if parts:
                        model_name = parts[0]
                        available_models.append(model_name)
            
            logger.debug(f"Available models: {available_models}")
            
            # verify if the model is available
            self._available = any(
                self.model_name in model.lower() 
                or model.lower().startswith(self.model_name)
                for model in available_models
                if model
            )
            
            if not self._available:
                logger.warning(
                    f"Model '{self.model_name}' not found. Available: {available_models}"
                )
                
        except subprocess.TimeoutExpired:
            logger.error("Ollama command timed out")
            self._available = False
        except FileNotFoundError:
            logger.error("Ollama not found in PATH. Please install Ollama.")
            self._available = False
        except Exception as e:
            logger.error(f"Error checking Ollama availability: {e}")
            self._available = False

    @property
    def is_available(self) -> bool:
        """Return whether Ollama is available."""
        return self._available

    def check_connection(self) -> Dict[str, Any]:
        """Verify that Ollama is running and model is available."""
        try:
            # Forcer une nouvelle vérification
            self._check_availability()
            
            if self._available:
                return {
                    "status": True,
                    "message": "Ollama connection successful"
                }
            else:
                return {
                    "status": False,
                    "message": f"Model '{self.model_name}' not found. Run: ollama pull {self.model_name}"
                }

        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return {
                "status": False,
                "message": f"Ollama connection failed: {str(e)}"
            }

    def generate_response(self, prompt: str, temperature: float = 0.2) -> str:
        """Generate response from Ollama model using subprocess."""
        if not self._available:
            return "⚠️ The LLM model is not available. Please check that Ollama is installed and that Mistral is downloaded.."

        try:
            # Utiliser subprocess pour appeler ollama run
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes max pour la réponse
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                error_msg = result.stderr.strip() if result.stderr else "Erreur inconnue"
                logger.error(f"Ollama run failed: {error_msg}")
                return f"❌ Erreur: {error_msg}"
                
        except subprocess.TimeoutExpired:
            return "⏱️ The model took too long to respond (120s)"
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"❌ Error: {str(e)}"

    def stream_response(self, prompt: str) -> Generator[str, None, None]:
        """Stream LLM answer token by token."""
        if not self._available:
            yield "⚠️The LLM model is not available."
            return

        try:
            process = subprocess.Popen(
                ["ollama", "run", self.model_name, prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            for line in process.stdout:
                if line.strip():
                    yield line.strip()
                    
            # verify if there is an error
            if process.stderr:
                error = process.stderr.read()
                if error:
                    logger.error(f"Stream error: {error}")
                    
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield f"❌ Erreur: {str(e)}"