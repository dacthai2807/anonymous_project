import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers.trainer import get_parameter_names
from transformers.trainer_utils import ALL_LAYERNORM_LAYERS

class LLaVALitModule(pl.LightningModule):
    def __init__(self, model, training_args, tokenizer=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = training_args

        # For saving model hyperparameters
        self.save_hyperparameters(ignore=["model", "tokenizer"])

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt_model = self.model
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        projector_parameters = [
            name for name, _ in opt_model.named_parameters() if "mm_projector" in name
        ]

        if self.args.mm_projector_lr is not None:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.mm_projector_lr,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.mm_projector_lr,
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = self.args.optimizer_cls_and_kwargs()

        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return optimizer
