"""
Document Router Agent.

Selects relevant documents before retrieval.
"""

import json
import logging
from typing import List

logger = logging.getLogger(__name__)


class DocumentRouter:
    """
    Decide which Markdown documents are relevant to a question.
    """

    def __init__(
        self,
        llm_client,
        file_catalog,
    ):
        self.llm_client = llm_client
        self.file_catalog = file_catalog

    def route(
        self,
        question: str,
    ) -> List[str]:
        """
        Return relevant Markdown filenames.

        The router guarantees that the returned value is:
            List[str]

        and that every filename exists in the catalog.
        """

        documents = self.file_catalog.get_documents()

        if not documents:
            return []

        available_files = {
            doc["file"]
            for doc in documents
            if doc.get("file")
        }

        catalog_text = "\n".join(
            [
                f"""
FILE: {doc['file']}
CATEGORY: {doc['category']}
DESCRIPTION:
{doc['description']}
"""
                for doc in documents
            ]
        )

        prompt = f"""
You are a document router.

Choose the most relevant Markdown documents
for answering the user's question.

Available documents:

{catalog_text}

User question:

{question}

Return ONLY a JSON list of filenames.

Example:

[
    "metrics.md",
    "validation.md"
]

Do not return explanations.
Do not return a JSON object.
Return only the JSON list.
"""

        try:
            response = self.llm_client.generate_response(
                prompt
            )

            selected = json.loads(response)

            # -------------------------------------------------
            # Validate top-level JSON type
            # -------------------------------------------------

            if not isinstance(selected, list):
                raise ValueError(
                    "Router response is not a JSON list."
                )

            # -------------------------------------------------
            # Validate each filename
            # -------------------------------------------------

            valid_selected = []

            for filename in selected:

                if not isinstance(filename, str):
                    continue

                filename = filename.strip()

                if filename in available_files:
                    valid_selected.append(filename)

            # Remove duplicates while preserving order.
            valid_selected = list(
                dict.fromkeys(valid_selected)
            )

            # -------------------------------------------------
            # If JSON was valid but no known file was selected,
            # fallback to all available files.
            # -------------------------------------------------

            if not valid_selected:
                logger.warning(
                    "Router returned no valid files. "
                    "Falling back to all documents."
                )

                return list(available_files)

            return valid_selected

        except Exception as e:

            logger.warning(
                "Document router failed: %s. "
                "Falling back to all documents.",
                e,
            )

            return list(available_files)