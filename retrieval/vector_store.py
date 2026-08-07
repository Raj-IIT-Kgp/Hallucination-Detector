import torch
import faiss
import numpy as np

from embeddings.embedder import create_embedding


class VectorStore:

    def __init__(self):

        self.documents = []

        self.index = None


    def add_documents(self, documents):

        self.documents = documents


        vectors = []

        for doc in documents:

            vector = create_embedding(doc)

            vectors.append(vector)


        vectors = np.array(vectors).astype("float32")


        dimension = vectors.shape[1]


        self.index = faiss.IndexFlatL2(
            dimension
        )


        self.index.add(vectors)



    def search(self, query, k=2):

        query_vector = create_embedding(query)

        query_vector = np.array(
            [query_vector]
        ).astype("float32")


        distances, indices = self.index.search(
            query_vector,
            k
        )


        results=[]


        for idx in indices[0]:

            results.append(
                self.documents[idx]
            )


        return results