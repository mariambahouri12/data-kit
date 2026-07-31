"""
FAISS vector storage module.
"""


import os
import pickle

import faiss



class VectorStore:


    def __init__(
        self,
        index_path="storage/rag_index/index.faiss",
        metadata_path="storage/rag_index/metadata.pkl"
    ):

        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index = None
        self.metadata = []



    def create(
        self,
        embeddings,
        documents
    ):
        """
        Create FAISS index.
        """


        dimension = embeddings.shape[1]


        self.index = faiss.IndexFlatIP(
            dimension
        )


        self.index.add(
            embeddings
        )


        self.metadata = documents



    def save(self):

        os.makedirs(
            os.path.dirname(self.index_path),
            exist_ok=True
        )


        faiss.write_index(
            self.index,
            self.index_path
        )


        with open(
            self.metadata_path,
            "wb"
        ) as file:

            pickle.dump(
                self.metadata,
                file
            )



    def load(self):

        self.index = faiss.read_index(
            self.index_path
        )


        with open(
            self.metadata_path,
            "rb"
        ) as file:

            self.metadata = pickle.load(file)



    def search(
        self,
        query_embedding,
        top_k=3
    ):
        """
        Retrieve nearest documents.
        """


        scores, indices = self.index.search(
            query_embedding,
            top_k
        )


        results = []


        for index in indices[0]:

            results.append(
                self.metadata[index]
            )


        return results