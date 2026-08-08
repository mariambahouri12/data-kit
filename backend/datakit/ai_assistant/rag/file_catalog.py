"""
File catalog for RAG routing.

Creates a lightweight index of available Markdown documents.
"""

from pathlib import Path
from typing import List, Dict


class FileCatalog:
    """
    Manage knowledge-base Markdown document metadata.
    """

    SUPPORTED_EXTENSION = ".md"

    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = Path(
            knowledge_base_path
        )
        self.documents: List[Dict] = []

    def build(self) -> List[Dict]:
        """
        Scan the knowledge base and create a document catalog.
        """

        self.documents = []

        if not self.knowledge_base_path.exists():
            return self.documents

        for file_path in self.knowledge_base_path.rglob("*.md"):

            if not file_path.is_file():
                continue

            category = file_path.parent.name

            try:
                content = file_path.read_text(
                    encoding="utf-8"
                )

            except Exception:
                content = ""

            self.documents.append(
                {
                    "file": file_path.name,
                    "path": str(file_path),
                    "category": category,
                    "description": self._generate_description(
                        content
                    ),
                }
            )

        return self.documents

    def _generate_description(
        self,
        content: str,
    ) -> str:
        """
        Generate a short description from the beginning
        of the Markdown document.
        """

        lines = content.split("\n")

        text = " ".join(
            [
                line.strip()
                for line in lines[:5]
                if line.strip()
            ]
        )

        return text[:300]

    def get_documents(self) -> List[Dict]:
        """
        Return the current document catalog.
        """

        return self.documents