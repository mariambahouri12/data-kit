"""
Load Markdown documents for RAG.
"""

from pathlib import Path
from typing import List, Dict, Any


class DocumentLoader:
    """
    Load Markdown documents from the knowledge base.
    """

    SUPPORTED_EXTENSION = ".md"

    def __init__(self, knowledge_base_path):
        self.base_path = Path(knowledge_base_path)

    def load(
        self,
        selected_files: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Load selected Markdown files.

        Args:
            selected_files:
                List of Markdown filenames.

        Returns:
            List of document dictionaries.
        """

        documents = []

        if not self.base_path.exists():
            return documents

        selected_set = set(selected_files or [])

        for file_path in self.base_path.rglob("*.md"):

            if file_path.name not in selected_set:
                continue

            try:
                content = file_path.read_text(
                    encoding="utf-8"
                )

                documents.append(
                    {
                        "content": content,
                        "source": file_path.name,
                        "category": file_path.parent.name,
                    }
                )

            except OSError:
                continue

        return documents