from contextlib import asynccontextmanager
from pathlib import Path
import io
import time

import mlflow.pytorch
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from PIL import Image
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from sneakers_hse.data.utils.s3_tools import S3Client
from sneakers_hse.inference.detection_model import YOLODetector
from sneakers_hse.inference.embedding_model import DINOv2Embedder
from sneakers_hse.inference.vector_store import VectorStore
from sneakers_hse.inference.logger import setup_logger

from db.models import Base, QueryLog
from db.session import engine, SessionLocal


logger = setup_logger()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    project_root_path: Path
    aws_access_key: str
    aws_secret_key: str
    embeddings_s3_path: str = "chroma_db"
    yolo_preprocessed_dataset_prefix: str = ""
    prod_execution_flg: str = "0"
    mlflow_tracking_uri: str = "http://103.76.55.124:5050"
    triplet_model_alias: str = "models:/triplet_loss@PRD"


settings = Settings()


# --- Response schemas ---

class SearchHit(BaseModel):
    ids: list[str]
    distances: list[float]
    metadatas: list[dict]
    image_urls: list[str]


class SearchResponse(BaseModel):
    results: list[SearchHit]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)

    s3 = S3Client(
        aws_access_key_id=settings.aws_access_key,
        aws_secret_access_key=settings.aws_secret_key,
    )

    if settings.prod_execution_flg == "1":
        logger.info("Downloading chroma_db from S3...")
        s3.download_folder_from_s3_parallel(
            bucket_name="sneakers-hse-images-test",
            s3_prefix=settings.embeddings_s3_path,
            local_folder=str(settings.project_root_path / "chroma_db"),
            max_workers=10,
        )

    app.state.s3 = s3
    app.state.detector = YOLODetector(
        str(settings.project_root_path / "models/yolov8n-clothing-detection.pt")
    )
    app.state.embedder = DINOv2Embedder()
    app.state.vector_store = VectorStore(
        persist_dir=str(settings.project_root_path / "chroma_db")
    )

    # MLflow needs standard boto3 env var names and Yandex S3 endpoint
    import os
    os.environ["AWS_ACCESS_KEY_ID"] = settings.aws_access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_key
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net")

    logger.info(f"Loading triplet model from MLflow: {settings.triplet_model_alias}")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    triplet_model = mlflow.pytorch.load_model(settings.triplet_model_alias)
    triplet_model.eval()
    app.state.triplet_model = triplet_model
    logger.info("Triplet model loaded")

    (settings.project_root_path / "tmp").mkdir(exist_ok=True)

    yield


app = FastAPI(title="Sneakers HSE Search", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search(image: UploadFile = File(...)):
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(400, "Only JPEG/PNG images are supported")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        logger.exception("Image decoding failed")
        raise HTTPException(400, "Invalid image file")

    t0 = time.perf_counter()

    try:
        bboxes = app.state.detector.detect(pil_image)
        logger.info(f"Detected {len(bboxes)} objects")
    except Exception:
        logger.exception("YOLO detection failed")
        raise HTTPException(500, "Detection error")

    results: list[SearchHit] = []
    for bbox in bboxes.values():
        # DINOv2 embedding
        dinov2_emb = app.state.embedder.encode_batch([bbox])  # (1, 768)

        # Triplet model refinement
        with torch.no_grad():
            emb_tensor = torch.tensor(dinov2_emb, dtype=torch.float32)
            refined_emb = app.state.triplet_model(emb_tensor).numpy()

        raw = app.state.vector_store.search(refined_emb[0])

        image_urls = [
            app.state.s3.client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": "sneakers-hse-images-test",
                    "Key": f"{settings.yolo_preprocessed_dataset_prefix}/{img_id}",
                },
                ExpiresIn=3600,
            )
            for img_id in raw["ids"][0]
        ]

        results.append(SearchHit(
            ids=raw["ids"][0],
            distances=raw["distances"][0],
            metadatas=raw["metadatas"][0],
            image_urls=image_urls,
        ))

    latency_ms = (time.perf_counter() - t0) * 1000

    _log_query(image.filename, results, latency_ms)

    return SearchResponse(results=results, latency_ms=latency_ms)


@app.post("/load_image")
async def load_image(path: str):
    tmp_dir = settings.project_root_path / "tmp"
    local_path = str(tmp_dir / Path(path).name)

    app.state.s3._download_one(
        bucket_name="sneakers-hse-images-test",
        s3_key=str(Path(settings.yolo_preprocessed_dataset_prefix) / path),
        local_path=local_path,
    )
    return FileResponse(local_path, media_type="image/jpeg")


def _log_query(filename: str | None, results: list[SearchHit], latency_ms: float):
    try:
        first = results[0] if results else None
        ids = first.ids if first else []
        db = SessionLocal()
        try:
            db.add(QueryLog(
                image_name=filename,
                top1=ids[0] if len(ids) > 0 else None,
                top2=ids[1] if len(ids) > 1 else None,
                top3=ids[2] if len(ids) > 2 else None,
                latency=latency_ms / 1000,
            ))
            db.commit()
        finally:
            db.close()
    except Exception:
        logger.warning("Failed to write QueryLog", exc_info=True)
