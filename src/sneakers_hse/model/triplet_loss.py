import sys
import numpy as np
import polars as pl
import pytorch_lightning as lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity


class EmbeddingDataset(Dataset):
    def __init__(self, df: pl.DataFrame):
        self.embeddings = torch.tensor(df["embedding"].to_list(), dtype=torch.float32)
        self.labels_str = df["class"].to_list()

        unique_labels = sorted(set(self.labels_str))
        self.label2idx = {l: i for i, l in enumerate(unique_labels)}
        self.labels = torch.tensor([self.label2idx[l] for l in self.labels_str])

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class LitTripletModel(lightning.LightningModule):
    def __init__(self, input_dim=768, embedding_dim=768, lr=1e-3, margin=0.2, dropout=0.3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, embedding_dim),
        )

        distance = CosineSimilarity()
        self.loss_fn = losses.TripletMarginLoss(margin=margin, distance=distance)
        self.miner = miners.TripletMarginMiner(margin=margin, distance=distance, type_of_triplets="semihard")

    def forward(self, x):
        return F.normalize(x + self.model(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        projected = self(x)
        hard_pairs = self.miner(projected, y)
        loss = self.loss_fn(projected, y, hard_pairs)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        projected = self(x)
        hard_pairs = self.miner(projected, y)
        loss = self.loss_fn(projected, y, hard_pairs)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class ValRetrievalMetricsCallback(lightning.Callback):
    def __init__(self, val_df: pl.DataFrame, batch_size: int = 64, check_every_n_epochs: int = 1):
        from sneakers_hse.metrics import get_neighbors
        self._get_neighbors = get_neighbors
        self.dataset = EmbeddingDataset(val_df)
        self.labels_str = np.array(val_df["class"].to_list())
        self.ids = [str(i) for i in range(len(self.labels_str))]
        self.batch_size = batch_size
        self.check_every_n_epochs = check_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        import chromadb

        if (trainer.current_epoch + 1) % self.check_every_n_epochs != 0:
            return

        dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
        pl_module.eval()
        device = pl_module.device

        parts = []
        for x, _ in dataloader:
            with torch.no_grad():
                parts.append(pl_module(x.to(device)).cpu().numpy())
        embeddings = np.vstack(parts)

        client = chromadb.Client()
        collection = client.get_or_create_collection("val_retrieval", metadata={"hnsw:space": "cosine"})
        for i in range(0, len(embeddings), 5000):
            collection.add(
                ids=self.ids[i:i + 5000],
                embeddings=embeddings[i:i + 5000],
                metadatas=[{"class": c} for c in self.labels_str[i:i + 5000].tolist()]
            )
        neighbors = self._get_neighbors(collection, embeddings, k=10)
        client.delete_collection("val_retrieval")

        metrics = {}
        for k in [1, 5, 10]:
            hits = float(np.mean([self.labels_str[i] in neighbors[i][:k] for i in range(len(self.labels_str))]))
            precision = float(np.mean([(neighbors[i][:k] == self.labels_str[i]).sum() / k for i in range(len(self.labels_str))]))
            metrics[f"val_hit@{k}"] = hits
            metrics[f"val_precision@{k}"] = precision

        if trainer.logger:
            trainer.logger.log_metrics(metrics, step=trainer.current_epoch)

        sys.__stdout__.write(
            f"[Epoch {trainer.current_epoch}] "
            f"Hit@1={metrics['val_hit@1']:.4f}  "
            f"Hit@5={metrics['val_hit@5']:.4f}  "
            f"Hit@10={metrics['val_hit@10']:.4f}\n"
        )
        sys.__stdout__.flush()
