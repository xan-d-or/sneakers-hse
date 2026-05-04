import polars as pl
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, df: pl.DataFrame):

        self.embeddings = torch.tensor(df["embedding"].to_list(), dtype=torch.float32)
        self.labels_str = df["class"].to_list()

        # label → int
        unique_labels = sorted(set(self.labels_str))
        self.label2idx = {l: i for i, l in enumerate(unique_labels)}
        self.labels = torch.tensor([self.label2idx[l] for l in self.labels_str])

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
    
