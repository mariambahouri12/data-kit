"""
Document preprocessing for the knowledge base.
"""

import re

from .document_loader import LoadedDocument


class DocumentProcessor:
    """Normalize raw knowledge-base documents."""

    def process(
        self,
        document: LoadedDocument,
    ) -> LoadedDocument:
        """Clean document content."""

        content = document.content

        content = content.replace(
            "\r\n",
            "\n",
        )

        content = re.sub(
            r"\n{3,}",
            "\n\n",
            content,
        )

        content = "\n".join(
            line.rstrip()
            for line in content.splitlines()
        ).strip()

        return LoadedDocument(
            document_id=document.document_id,
            name=document.name,
            content=content,
            source=document.source,
        )