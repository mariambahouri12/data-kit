"""
Ollama LLM client.
"""

import requests


class OllamaClient:
    """Minimal client for a local Ollama server."""

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
    ) -> str:
        """Generate a response from Ollama."""

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        return data["response"].strip()