from litdata import StreamingDataset
import torch

class StreamingImageDataset(StreamingDataset):
    def __init__(self, input_dir, transform=None):
        super().__init__(input_dir=input_dir)
        self.transform = transform

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        img = sample["image"]
        label = sample["label"]

        if self.transform:
            img = self.transform(image=img)["image"]

        label = torch.tensor(label, dtype=torch.long)

        return img, label
