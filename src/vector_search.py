import faiss
import numpy as np

class VectorSearch:

    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)

    def add(self, embeddings):
        self.index.add(np.array(embeddings))

    def search(self, query, k=3):
        return self.index.search(np.array([query]), k)