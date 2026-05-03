from litdata import StreamingDataset
from torch.utils.data import Dataset
import torch

import PIL.Image as Image
import PIL
import numpy


class LitDataImageDataset(Dataset):
    def __init__(self, input_dir, label2idx, transform=None, train=True,
                 cache_dir=None, **kwargs):
        self.ds = StreamingDataset(input_dir, shuffle=train, drop_last=train, 
                                   cache_dir=cache_dir, **kwargs)
        self.transform = transform
        self.label2idx = label2idx
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        image = item["image"]
        label = item["label"].split("/")[-1]  # извлекаем имя класса из пути
        label = self.label2idx[label]

        # Это надо для albumentations
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()

        image = self.transform(image=image)["image"]

        return image, label

