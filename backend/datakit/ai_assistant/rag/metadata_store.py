"""
Metadata management for vector documents.
"""


class MetadataStore:
    """
    Normalize and enrich document metadata.
    """

    def enrich(self, documents):
        """
        Enrich documents while preserving their metadata.

        Args:
            documents:
                List of document dictionaries.

        Returns:
            List of enriched documents.
        """

        if not documents:
            return []

        enriched = []

        for idx, doc in enumerate(documents):

            if isinstance(doc, dict):

                metadata = doc.copy()

                metadata.setdefault(
                    "content",
                    "",
                )

                metadata.setdefault(
                    "source",
                    "unknown",
                )

                metadata.setdefault(
                    "category",
                    "unknown",
                )

                metadata["id"] = idx

                enriched.append(metadata)

            else:

                enriched.append(
                    {
                        "id": idx,
                        "content": str(doc),
                        "source": "unknown",
                        "category": "unknown",
                    }
                )

        return enriched