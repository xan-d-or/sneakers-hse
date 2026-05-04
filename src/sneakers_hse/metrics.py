import numpy as np

def get_neighbors(collection, embeddings, k=10, batch_size=10):
    all_results = []
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i+batch_size]
        results = collection.query(
            query_embeddings=batch,
            n_results=k + 1  # +1 чтобы убрать self
        )
        if i == 0:
            print(results)
        all_results.extend(results['metadatas'])
    return np.array([[neighbor['class'] for neighbor in query]
                     for query in all_results])[:, 1:]  # Убираю self-match