import logging
import re
import subprocess
from typing import Dict, Any, Generator

logger = logging.getLogger(__name__)

# FIX (codes ANSI dans la réponse) : `ollama run` émet des séquences de
# contrôle terminal (déplacement curseur, effacement de ligne) destinées
# à un vrai terminal interactif. Capturées via subprocess.run(), elles
# restent brutes dans stdout au lieu d'être interprétées -> polluent le
# texte final avec des "\x1b[2D\x1b[K" visibles. On les filtre avant de
# retourner la réponse.
_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub('', text)


class OllamaClient:
    """Client to interact with local Ollama LLM using subprocess."""

    def __init__(self, model_name: str = "mistral", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self._available = False
        self._check_availability()

    def _check_availability(self) -> None:
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=3
            )
            if result.returncode != 0:
                logger.warning("Ollama not installed or not in PATH")
                self._available = False
                return

            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=5
            )
            if result.returncode != 0:
                logger.warning(f"Ollama list failed: {result.stderr}")
                self._available = False
                return

            lines = result.stdout.strip().split("\n")
            available_models = []
            for line in lines[1:]:
                if line.strip():
                    parts = line.split()
                    if parts:
                        available_models.append(parts[0])

            self._available = any(
                self.model_name in model.lower() or model.lower().startswith(self.model_name)
                for model in available_models if model
            )

            if not self._available:
                logger.warning(f"Model '{self.model_name}' not found. Available: {available_models}")

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
        return self._available

    def check_connection(self) -> Dict[str, Any]:
        try:
            self._check_availability()
            if self._available:
                return {"status": True, "message": "Ollama connection successful"}
            return {
                "status": False,
                "message": f"Model '{self.model_name}' not found. Run: ollama pull {self.model_name}"
            }
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return {"status": False, "message": f"Ollama connection failed: {str(e)}"}

    def generate_response(self, prompt: str, temperature: float = 0.2) -> str:
        if not self._available:
            return "⚠️ The LLM model is not available. Please check that Ollama is installed and that Mistral is downloaded."

        try:
            process = subprocess.run(
                ["ollama", "run", self.model_name],
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120
            )

            if process.returncode == 0:
                return _strip_ansi(process.stdout).strip()
            else:
                error_msg = process.stderr.strip() if process.stderr else "Unknown error"
                logger.error(f"Ollama run failed: {error_msg}")
                return f"❌ Error: {error_msg}"

        except subprocess.TimeoutExpired:
            return "⏱️ The model took too long to respond (120s)"
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"❌ Error: {str(e)}"

    def stream_response(self, prompt: str) -> Generator[str, None, None]:
        if not self._available:
            yield "⚠️ The LLM model is not available."
            return

        try:
            process = subprocess.Popen(
                ["ollama", "run", self.model_name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1
            )

            process.stdin.write(prompt)
            process.stdin.close()

            for line in process.stdout:
                clean_line = _strip_ansi(line).strip()
                if clean_line:
                    yield clean_line

            if process.stderr:
                error = process.stderr.read()
                if error:
                    logger.error(f"Stream error: {error}")

        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield f"❌ Error: {str(e)}"