import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path


class ImageDataset(Dataset):
    def __init__(self, base_path, images_path, labels, class_to_idx, augmenter=None):
        self.base_path = Path(base_path)
        self.images_path = images_path
        self.labels = labels
        self.augmenter = augmenter
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_path = self.base_path / self.images_path[idx]
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        if self.augmenter:
            img = self.augmenter(image=img)["image"]

        label_str = self.labels[idx]
        label = self.class_to_idx[label_str]
        label = torch.tensor(label, dtype=torch.long)

        return img, label

