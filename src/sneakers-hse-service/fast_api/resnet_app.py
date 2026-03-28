import sys
sys.path.append('../../..')

from fastapi import FastAPI, UploadFile, File, HTTPException, Path, status
import numpy as np
from pathlib import Path
import pickle
import cv2
import torch
from src.model.resnet_18 import LitResNet18
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Загружаем модель и маппинг в человекочитаемые названия классов
# models_path = Path("../../../models")
models_path = Path(__file__).parent.parent.parent.parent.resolve() / 'models'
with open(models_path / 'resnet18_new_dataset0201_class_names.json', 'r', encoding='utf-8') as f:
    idx_to_class = json.load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LitResNet18(num_classes=len(idx_to_class))
model.load_state_dict(torch.load(models_path / "resnet18_new_dataset0201.pth", map_location=device))
model.to(device)
model.eval()


def preprocess_image(image: np.ndarray):
    augmenter = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])
    return augmenter(image=image)['image']

def infer_model(image: np.ndarray):
    input_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():  # Faster inference
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = idx_to_class[str(probs.argmax().item())]
    return pred_class


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
    if image.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='bad request'
        )

    contents = await image.read()

    image_array = cv2.imdecode(np.asarray(bytearray(contents), dtype=np.uint8), cv2.IMREAD_COLOR)
    image_array = preprocess_image(image_array)

    try:
        predicted_class = str(infer_model(image_array))
    except:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Модель не смогла обработать данные'
        )
    return {
        "filename": image.filename,
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
