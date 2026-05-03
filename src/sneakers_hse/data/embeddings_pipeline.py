from pathlib import Path
from PIL import Image
from tqdm import tqdm

from sneakers_hse.inference.embedding_model import DinoEmbedder
from sneakers_hse.inference.vector_store import VectorStore

def build_index(data_dir: Path):
    model = DinoEmbedder()
    store = VectorStore()

    embeddings = []
    ids = []
    metadatas = []

    for cls_dir in data_dir.iterdir():
        for img_path in cls_dir.glob("*.jpg"):
            image = Image.open(img_path).convert("RGB")

            emb = model.encode(image)

            embeddings.append(emb)
            ids.append(str(img_path))
            metadatas.append({
                "class": cls_dir.name,
                "path": str(img_path)
            })

    store.add(embeddings, ids, metadatas)