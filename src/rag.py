from embeddings import EmbeddingModel

class SimpleRAG:

    def __init__(self):
        self.embedder = EmbeddingModel()

    def embed(self, docs):
        return self.embedder.encode(docs)