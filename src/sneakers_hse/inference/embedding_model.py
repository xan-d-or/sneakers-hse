import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import numpy as np

    
class DINOv2Embedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        self.model.eval()
    def encode_batch(self, images: list[Image.Image]):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_tokens = outputs.last_hidden_state[:, 0, :]
        cls_tokens = cls_tokens.cpu().numpy()
        norms = np.linalg.norm(cls_tokens, axis=1, keepdims=True)
        return cls_tokens / norms