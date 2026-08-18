"""
Document chunking.
"""

from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


@dataclass
class DocumentChunk:
    """A chunk extracted from a document."""

    chunk_id: str
    document_id: str
    document_name: str
    content: str
    source: str


class TextChunker:
    """
    Split documents by Markdown headers and then enforce a maximum
    token size, with a token overlap between adjacent chunks to
    preserve context across cut points.
    """

    HEADERS_TO_SPLIT_ON = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 0,
        encoding_name: str = "cl100k_base",
    ) -> None:

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )

        self.size_splitter = (
            RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=encoding_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def split(self, document) -> list[DocumentChunk]:
        """
        Split a document into chunks.

        The same chunks are later indexed into:
        - Qdrant for semantic search
        - Elasticsearch for lexical/BM25 search
        """

        text = document.content

        if not text:
            return []

        sections = self.header_splitter.split_text(text)

        chunks: list[DocumentChunk] = []
        chunk_number = 0

        for section in sections:

            header_path = " > ".join(
                str(value) for value in section.metadata.values()
            )

            content = section.page_content.strip()

            if not content:
                continue

            if self._count_tokens(content) > self.chunk_size:
                parts = self.size_splitter.split_text(content)
            else:
                parts = [content]

            for part in parts:

                part = part.strip()

                if not part:
                    continue

                prefixed_content = (
                    f"{header_path}\n{part}" if header_path else part
                )

                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document.document_id}:{chunk_number}",
                        document_id=document.document_id,
                        document_name=document.name,
                        content=prefixed_content,
                        source=document.source,
                    )
                )

                chunk_number += 1

        return chunks