"""
Knowledge-base document loading.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LoadedDocument:
    """Raw document loaded from the knowledge base."""

    document_id: str
    name: str
    content: str
    source: str


class DocumentLoader:
    """Load Markdown documents from a directory."""

    SUPPORTED_EXTENSIONS = {".md"}

    def __init__(
        self,
        knowledge_base_path: str,
    ) -> None:
        self.root = Path(
            knowledge_base_path
        )

    def load(self) -> list[LoadedDocument]:
        """Load all supported documents."""

        if not self.root.exists():
            raise FileNotFoundError(
                f"Knowledge base not found: {self.root}"
            )

        documents = []

        for path in sorted(
            self.root.rglob("*")
        ):
            if (
                path.is_file()
                and path.suffix.lower()
                in self.SUPPORTED_EXTENSIONS
            ):
                documents.append(
                    LoadedDocument(
                        document_id=path.stem,
                        name=path.name,
                        content=path.read_text(
                            encoding="utf-8"
                        ),
                        source=str(path),
                    )
                )

        return documents