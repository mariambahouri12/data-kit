"""
Retriever module.

Responsible for finding relevant knowledge
documents for a user query.
"""


import numpy as np



class Retriever:


    def __init__(
        self,
        embedding_model,
        vector_store
    ):

        self.embedding_model = embedding_model

        self.vector_store = vector_store



    def retrieve(
        self,
        query,
        top_k=3
    ):
        """
        Retrieve relevant documents.
        """


        query_embedding = (
            self.embedding_model
            .encode([query])
        )


        query_embedding = np.array(
            query_embedding
        ).astype("float32")



        documents = (
            self.vector_store
            .search(
                query_embedding,
                top_k
            )
        )


        return documents