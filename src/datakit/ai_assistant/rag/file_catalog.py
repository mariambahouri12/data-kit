"""
File catalog for RAG routing.

Creates a lightweight index of available documents.
"""

import os
from pathlib import Path
from typing import List, Dict

class FileCatalog:
    """
    Manage knowledge base document metadata.
    """

    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.documents = []

    def build(self) -> List[Dict]:
        """
        Scan knowledge base and create catalog.
        Supports .md, .txt, .pdf, .csv files.
        """

        self.documents = []

        # Supported file extensions
        supported_extensions = {".md", ".txt", ".pdf", ".csv"}

        for file_path in self.knowledge_base_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                category = file_path.parent.name

                try:
                    # Try to read text content
                    content = file_path.read_text(
                        encoding="utf-8"
                    )
                except Exception:
                    # For binary files like PDF, just use filename as content
                    content = f"File: {file_path.name}"

                self.documents.append(
                    {
                        "file": file_path.name,
                        "path": str(file_path),
                        "category": category,
                        "description": self._generate_description(content)
                    }
                )

        return self.documents

    def _generate_description(self, content: str):
        """
        Generate short document description.
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

    def get_documents(self):
        return self.documents