"""
Rebuilds ChromaDB: replaces raw DINOv2 embeddings with triplet-refined ones.

Usage (from project root):
    python scripts/rebuild_chroma_triplet.py
"""
import os
from pathlib import Path
import numpy as np
import torch
import chromadb
import mlflow.pytorch

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://103.76.55.124:5050")
MODEL_ALIAS = "models:/triplet_loss@PRD"
CHROMA_DIR = os.getenv("CHROMA_DIR", "/app/chroma_db")
SRC_COLLECTION = "embeddings"  # raw DINOv2
DST_COLLECTION = "embeddings"  # overwrite in place with triplet embeddings
BATCH_SIZE = 256

print(f"Loading triplet model from {MLFLOW_TRACKING_URI} ...")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pytorch.load_model(MODEL_ALIAS)
model.eval()
print("Model loaded.")

client = chromadb.PersistentClient(path=CHROMA_DIR)
src = client.get_collection(SRC_COLLECTION)
total = src.count()
print(f"Source collection '{SRC_COLLECTION}': {total} items")

# Read all embeddings in batches
all_ids, all_embeddings, all_metadatas, all_documents = [], [], [], []
offset = 0
while offset < total:
    batch = src.get(
        limit=BATCH_SIZE,
        offset=offset,
        include=["embeddings", "metadatas", "documents"],
    )
    all_ids.extend(batch["ids"])
    all_embeddings.extend(batch["embeddings"])
    all_metadatas.extend(batch["metadatas"] or [None] * len(batch["ids"]))
    all_documents.extend(batch["documents"] or [None] * len(batch["ids"]))
    offset += len(batch["ids"])
    print(f"  read {offset}/{total}")

print("Transforming embeddings through triplet model...")
emb_tensor = torch.tensor(all_embeddings, dtype=torch.float32)
refined_list = []
with torch.no_grad():
    for i in range(0, len(emb_tensor), BATCH_SIZE):
        chunk = emb_tensor[i : i + BATCH_SIZE]
        refined = model(chunk).numpy()
        refined_list.append(refined)
        print(f"  transformed {min(i + BATCH_SIZE, len(emb_tensor))}/{len(emb_tensor)}")
refined_all = np.concatenate(refined_list, axis=0)

# Delete old collection and recreate with triplet embeddings
print(f"Replacing collection '{DST_COLLECTION}' with triplet embeddings...")
try:
    client.delete_collection(DST_COLLECTION)
except Exception:
    pass
dst = client.create_collection(DST_COLLECTION, metadata={"hnsw:space": "cosine"})

for i in range(0, len(all_ids), BATCH_SIZE):
    batch_ids = all_ids[i : i + BATCH_SIZE]
    batch_embs = refined_all[i : i + BATCH_SIZE].tolist()
    batch_meta = all_metadatas[i : i + BATCH_SIZE]
    batch_docs = all_documents[i : i + BATCH_SIZE]
    dst.add(
        ids=batch_ids,
        embeddings=batch_embs,
        metadatas=[m or {} for m in batch_meta],
        documents=[d or "" for d in batch_docs],
    )
    print(f"  written {min(i + BATCH_SIZE, len(all_ids))}/{len(all_ids)}")

print(f"Done. Collection '{DST_COLLECTION}' now has {dst.count()} triplet embeddings.")
