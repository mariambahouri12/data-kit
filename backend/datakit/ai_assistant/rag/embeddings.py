"""
Embedding generation module.

Responsible for converting text documents
into numerical vectors.
"""

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper around Sentence Transformer models.
    """


    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):

        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )


    def encode(
        self,
        texts
    ):
        """
        Convert texts into embeddings.

        Args:
            texts:
                List of strings

        Returns:
            Embeddings matrix
        """

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings