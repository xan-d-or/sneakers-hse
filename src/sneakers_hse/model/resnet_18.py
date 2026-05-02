import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


class LitResNet18(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-3):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr

        self.val_preds = []
        self.val_targets = []

       
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    
    def forward(self, x):
        return self.model(x)

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = logits.argmax(dim=1)

        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(y.detach().cpu())

        self.log("val_batch_loss", loss, prog_bar=True)
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

        self.log("val_acc_micro", acc_micro, prog_bar=True, logger=True)
        self.log("val_acc_macro", acc_macro, prog_bar=True, logger=True)
        self.log("val_f1_micro", f1_micro, prog_bar=True, logger=True)
        self.log("val_f1_macro", f1_macro, prog_bar=True, logger=True)
        self.log("val_f1_weighted", f1_weighted, prog_bar=True, logger=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        
        logits = self(x)
        preds = logits.argmax(dim=1)
        
        return preds

