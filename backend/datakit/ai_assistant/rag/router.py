"""
Document Router Agent.

Selects relevant documents before retrieval.
"""

import json
import logging


logger = logging.getLogger(__name__)


class DocumentRouter:

    """
    Decide which documents are relevant.
    """


    def __init__(
        self,
        llm_client,
        file_catalog
    ):

        self.llm_client = llm_client
        self.file_catalog = file_catalog



    def route(
        self,
        question: str
    ):
        """
        Return selected documents.
        """


        documents = (
            self.file_catalog.get_documents()
        )


        if not documents:
            return []



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

Choose the most relevant documents
for answering the user question.

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

"""


        response = (
            self.llm_client
            .generate_response(prompt)
        )


        try:

            selected = json.loads(response)

            return selected


        except Exception:

            logger.warning(
                "Router failed, using all documents"
            )

            return [
                doc["file"]
                for doc in documents
            ]