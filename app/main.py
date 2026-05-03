from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import io
import os
from pathlib import Path
from dotenv import load_dotenv

from sneakers_hse.data.utils.s3_tools import S3Client

from sneakers_hse.inference.detection_model import YOLODetector
from sneakers_hse.inference.embedding_model import DinoEmbedder
from sneakers_hse.inference.vector_store import VectorStore
from sneakers_hse.inference.logger import setup_logger


logger = setup_logger()

load_dotenv()

PROJECT_ROOT_PATH = Path(os.getenv("PROJECT_ROOT_PATH")) 

s3 = S3Client(aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
              aws_secret_access_key=os.getenv("AWS_SECRET_KEY"))


if os.getenv("PROD_EXECUTION_FLG") == 1:
    s3.download_folder_from_s3_parallel(
        bucket_name='sneakers-hse-images-test',
        s3_prefix=os.getenv("EMBEDDINGS_S3_PATH"),
        local_folder=str(PROJECT_ROOT_PATH / 'chroma_db'),
        max_workers=10
    )

app = FastAPI()

detector = YOLODetector(str(PROJECT_ROOT_PATH / 
                            'models/yolov8n-clothing-detection.pt'))
embedder = DinoEmbedder()
vector_store = VectorStore()

@app.post("/search")
async def search(image: UploadFile = File(...)):
    if image.content_type not in ["image/jpeg", "image/png", 'image/jpg']:
        raise HTTPException(400, "Invalid image")
    
    try:
        contents = await image.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.error(f"Image decoding failed: {e}")
        raise HTTPException(400, "Invalid image file")
    
    try:
        bboxes = detector.detect(image)
        logger.info(f"Detected {len(bboxes)} objects")
    except Exception as e:
        logger.exception("YOLO detection failed")
        raise HTTPException(500, "Detection error")

    res = []
    # Если нашли несколько ббоксов с обувью, то для каждого находим ближайших соседейы
    for bbox in bboxes.values():
        embedding = embedder.encode(bbox)
        results = vector_store.search(embedding)
        logger.info(results)
        res.append(results)
    
    return {
        "results": res
    }


@app.post("/load_image")
async def load_image(path: str):
    local_path = str(PROJECT_ROOT_PATH / 'tmp')
    s3._download_one(
        bucket_name='sneakers-hse-images-test',
        s3_key=str(Path(os.getenv("YOLO_PREPROCESSED_DATASET_PREFIX")) / path),
        local_path=local_path
    )
    return FileResponse(local_path, media_type="image/jpeg")