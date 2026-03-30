import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm

from sklearn.metrics import accuracy_score, f1_score
import numpy as np


class LitClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name,
        num_classes,
        lr=1e-3,
        freeze_backbone=True,
        label_smoothing=0.1
    ):
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr

        self.val_preds = []
        self.val_targets = []

        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes
        )

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if not any(k in name for k in ["classifier", "fc", "head"]):
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = logits.argmax(dim=1)

        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(y.detach().cpu())

        self.log("val_loss", loss, prog_bar=True)

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

        self.log("val_acc_micro", acc_micro, prog_bar=True)
        self.log("val_acc_macro", acc_macro, prog_bar=True)
        self.log("val_f1_micro", f1_micro, prog_bar=True)
        self.log("val_f1_macro", f1_macro, prog_bar=True)
        self.log("val_f1_weighted", f1_weighted, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        logits = self(x)
        return logits.argmax(dim=1)
