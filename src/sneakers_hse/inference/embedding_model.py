import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DinoEmbedder:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant-imagenet1k-1-layer')
        self.model = AutoModel.from_pretrained('facebook/dinov2-giant-imagenet1k-1-layer')

    def encode(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_tokens: torch.Tensor = outputs.last_hidden_state[:, 0, :]
        return cls_tokens[0].numpy()