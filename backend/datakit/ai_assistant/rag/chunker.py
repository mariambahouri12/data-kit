"""
Document chunking.
"""

from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """A chunk extracted from a document."""

    chunk_id: str
    document_id: str
    document_name: str
    content: str
    source: str


class TextChunker:
    """Split documents into overlapping text chunks."""

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 100,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(
        self,
        document,
    ) -> list[DocumentChunk]:
        """Split a document into chunks."""

        text = document.content

        if not text:
            return []

        chunks = []
        start = 0
        chunk_number = 0

        while start < len(text):
            end = min(
                start + self.chunk_size,
                len(text),
            )

            content = text[start:end].strip()

            if content:
                chunks.append(
                    DocumentChunk(
                        chunk_id=(
                            f"{document.document_id}:"
                            f"{chunk_number}"
                        ),
                        document_id=document.document_id,
                        document_name=document.name,
                        content=content,
                        source=document.source,
                    )
                )

                chunk_number += 1

            if end >= len(text):
                break

            start = end - self.overlap

        return chunks