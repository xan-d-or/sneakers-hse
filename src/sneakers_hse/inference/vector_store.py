import chromadb

class VectorStore:
    def __init__(self, persist_dir="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("embeddings")

    def add(self, embeddings, ids, metadatas=None):
        self.collection.add(
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

    def search(self, embedding, k=5):
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k
        )
        return results