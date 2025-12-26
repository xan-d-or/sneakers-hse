from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Path, status
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
import numpy as np
import pickle
from enum import Enum
import aiofiles
import cv2

def extract_hog_features(image):
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    img_resized = cv2.resize(image, winSize)
    features = hog.compute(img_resized).flatten()
    return features


def infer_model(image: np.ndarray):

    with open("../../../models/baseline_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("../../../models/svc.pkl", "rb") as f:
        svc = pickle.load(f)

    hog_image = extract_hog_features(image)
    predicted_class = svc.predict(
        scaler.transform(hog_image.reshape(1, -1))
    )

    return predicted_class


# Initialize FastAPI application
# Инициализация приложения FastAPI
app = FastAPI(
    title="FastAPI For Sneaker Classification",
    description="Classification sneakers by image",
    version="1.0.0"
)

@app.post("/forward", status_code=status.HTTP_201_CREATED)
async def forward(
    image: UploadFile = File(...)
):  

    contents = await image.read()

    image_array = cv2.imdecode(np.asarray(bytearray(contents), dtype=np.uint8), cv2.IMREAD_COLOR)

    async with aiofiles.open(f"image.png", "wb") as f:
        await f.write(contents)

    predicted_class = str(infer_model(image_array)[0]) ## c взятием первого элемента и приведения к str это костыль конеш
    return {
        "filename": image.filename,
        "size": len(contents),
        "predicted_class": predicted_class
    }


if __name__ == "__main__":
    import uvicorn
    ## Запускать как uvicorn app:app --reload
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
