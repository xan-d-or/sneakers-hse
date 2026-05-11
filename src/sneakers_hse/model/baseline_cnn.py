import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


class LitCNN(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.val_preds = []
        self.val_targets = []

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = logits.argmax(dim=1)

        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(y.detach().cpu())

        self.log("val_batch_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds).numpy()
        targets = torch.cat(self.val_targets).numpy()

        self.val_preds.clear()
        self.val_targets.clear()

        acc_micro = accuracy_score(targets, preds)

        class_acc = []
        for cls in np.unique(targets):
            mask = targets == cls
            class_acc.append(accuracy_score(targets[mask], preds[mask]))
        acc_macro = np.mean(class_acc)

        f1_micro = f1_score(targets, preds, average="micro")
        f1_macro = f1_score(targets, preds, average="macro")
        f1_weighted = f1_score(targets, preds, average="weighted")

        self.log("val_acc_micro", acc_micro, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc_macro", acc_macro, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1_micro", f1_micro, prog_bar=True)
        self.log("val_f1_macro", f1_macro, prog_bar=True)
        self.log("val_f1_weighted", f1_weighted, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        
        logits = self(x)
        preds = logits.argmax(dim=1)
        
        return preds
