"""
Prompt construction for the DataKit assistant.
"""


class PromptBuilder:
    """Build prompts for the DataKit LLM."""

    def build(
        self,
        question: str,
        documents: list[dict],
        dataset_context: str,
    ) -> str:

        knowledge_context = self._format_documents(
            documents
        )

        return f"""
You are the DataKit AI assistant.

Answer the user's question using only the
provided context.

If the context does not contain enough information,
say that the available context is insufficient.

Do not invent facts.

### Knowledge Base

{knowledge_context}

### Dataset Context

{dataset_context}

### User Question

{question}

### Answer
""".strip()

    @staticmethod
    def _format_documents(
        documents: list[dict],
    ) -> str:

        if not documents:
            return "No relevant knowledge-base documents found."

        sections = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            sections.append(
                f"[Document {index}]\n"
                f"Source: {document.get('document_name', '')}\n"
                f"{document.get('content', '')}"
            )

        return "\n\n".join(sections)