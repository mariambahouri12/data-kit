"""
Metadata management for vector documents.
"""

class MetadataStore:

    def enrich(
        self,
        documents
    ):
        """
        Enrich documents with metadata while preserving all original fields.
        
        Args:
            documents: List of document dictionaries
                Each doc should have at least 'content'
        
        Returns:
            List of enriched documents with all original metadata + id
        """
        
        if not documents:
            return []
        
        enriched = []

        for idx, doc in enumerate(documents):
            # Handle different document formats
            if isinstance(doc, dict):
                # Create a copy of all metadata
                metadata = doc.copy()
                
                # Add or override id
                metadata["id"] = idx
                
                # Ensure required fields exist
                if "content" not in metadata:
                    metadata["content"] = ""
                if "source" not in metadata:
                    metadata["source"] = "unknown"
                if "category" not in metadata:
                    metadata["category"] = "unknown"
                
                enriched.append(metadata)
            else:
                # If document is a string
                enriched.append(
                    {
                        "id": idx,
                        "content": str(doc),
                        "source": "unknown",
                        "category": "unknown"
                    }
                )

        return enriched