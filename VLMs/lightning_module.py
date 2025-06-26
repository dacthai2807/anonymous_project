import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class VLMModule(pl.LightningModule):
    def __init__(self, model, tokenizer=None, lr=1e-4):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer  # optional
        self.lr = lr

    def forward(self, images, texts):
        return self.model(images, texts)

    def training_step(self, batch, batch_idx):
        images, texts, labels = batch
        outputs = self(images, texts)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, texts, labels = batch
        outputs = self(images, texts)
        loss = F.cross_entropy(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):
        return super().on_train_epoch_end()
    
    def on_validation_epoch_end(self):
        return super().on_validation_epoch_end()
    
    def on_test_epoch_end(self):
        return super().on_test_epoch_end()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
