import pytorch_lightning as pl
import torch
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, StepLR
import re

class LLaVALitModule(pl.LightningModule):
    def __init__(self, model, training_args, tokenizer=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = training_args
        self.optimizer = None

        # Save hyperparameters except model and tokenizer
        self.save_hyperparameters(ignore=["model", "tokenizer"])

    def forward(self, **batch):
        return self.model(**batch)

    def _shared_step(self, batch, step_name):
        outputs = self(**batch)
        loss = getattr(outputs, "loss", outputs.get("loss"))
        self.log(f"{step_name}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def get_parameter_names(self, model, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        Adapted from transformers.trainer.get_parameter_names
        """
        result = []
        for name, child in model.named_modules():
            result += [
                f"{name}.{n}"
                for n, p in child.named_parameters(recurse=False)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        return result

    def get_layernorm_layer_types(self):
        """
        Get all LayerNorm-like layer types.
        Adapted from transformers ALL_LAYERNORM_LAYERS
        """
        layernorm_types = []
        
        # Standard PyTorch LayerNorm types
        if hasattr(torch.nn, 'LayerNorm'):
            layernorm_types.append(torch.nn.LayerNorm)
        if hasattr(torch.nn, 'GroupNorm'):
            layernorm_types.append(torch.nn.GroupNorm)
        if hasattr(torch.nn, 'InstanceNorm1d'):
            layernorm_types.append(torch.nn.InstanceNorm1d)
        if hasattr(torch.nn, 'InstanceNorm2d'):
            layernorm_types.append(torch.nn.InstanceNorm2d)
        if hasattr(torch.nn, 'InstanceNorm3d'):
            layernorm_types.append(torch.nn.InstanceNorm3d)
        if hasattr(torch.nn, 'BatchNorm1d'):
            layernorm_types.append(torch.nn.BatchNorm1d)
        if hasattr(torch.nn, 'BatchNorm2d'):
            layernorm_types.append(torch.nn.BatchNorm2d)
        if hasattr(torch.nn, 'BatchNorm3d'):
            layernorm_types.append(torch.nn.BatchNorm3d)
            
        # Try to import transformers-specific norm layers if available
        try:
            from transformers.models.t5.modeling_t5 import T5LayerNorm
            layernorm_types.append(T5LayerNorm)
        except ImportError:
            pass
            
        try:
            from transformers.models.llama.modeling_llama import LlamaRMSNorm
            layernorm_types.append(LlamaRMSNorm)
        except ImportError:
            pass
            
        return layernorm_types

    def get_optimizer_class_and_kwargs(self, args):
        """
        Get optimizer class and kwargs based on training arguments.
        Replaces Trainer.get_optimizer_cls_and_kwargs
        """
        # Default optimizer settings
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "lr": getattr(args, 'learning_rate', 5e-5),
            "eps": getattr(args, 'adam_epsilon', 1e-8),
            "betas": (getattr(args, 'adam_beta1', 0.9), getattr(args, 'adam_beta2', 0.999)),
        }
        
        # Handle different optimizer types
        optim_name = getattr(args, 'optim', 'adamw_torch').lower()
        
        if 'adamw' in optim_name:
            optimizer_cls = AdamW
        elif 'adam' in optim_name:
            optimizer_cls = Adam
        elif 'sgd' in optim_name:
            optimizer_cls = SGD
            optimizer_kwargs = {
                "lr": getattr(args, 'learning_rate', 5e-5),
                "momentum": getattr(args, 'momentum', 0.9),
            }
            
        return optimizer_cls, optimizer_kwargs

    def configure_optimizers(self):
        """
        Setup the optimizer.
        
        This method replaces the Trainer-dependent optimizer configuration
        with a pure PyTorch Lightning implementation.
        """
        opt_model = self.model

        if self.optimizer is None:
            # Get LayerNorm layer types
            layernorm_layer_types = self.get_layernorm_layer_types()
            
            # Get parameter names that should have weight decay
            decay_parameters = self.get_parameter_names(opt_model, layernorm_layer_types)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            # Check if we have mm_projector specific learning rate
            if hasattr(self.args, 'mm_projector_lr') and self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": getattr(self.args, 'weight_decay', 0.01),
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
                        "weight_decay": getattr(self.args, 'weight_decay', 0.01),
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
                        "weight_decay": getattr(self.args, 'weight_decay', 0.01),
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_class_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        # Configure scheduler if specified
        scheduler_config = None
        if hasattr(self.args, 'lr_scheduler_type') and self.args.lr_scheduler_type:
            scheduler = self.get_scheduler()
            if scheduler:
                scheduler_config = {
                    "scheduler": scheduler,
                    "interval": "step",  # or "epoch"
                    "frequency": 1,
                }

        if scheduler_config:
            return {"optimizer": self.optimizer, "lr_scheduler": scheduler_config}
        else:
            return self.optimizer

    def get_scheduler(self):
        """
        Create learning rate scheduler based on training arguments.
        """
        if not hasattr(self.args, 'lr_scheduler_type'):
            return None
            
        scheduler_type = getattr(self.args, 'lr_scheduler_type', None)
        if not scheduler_type:
            return None
            
        # Get total steps for schedulers that need them
        total_steps = getattr(self.args, 'max_steps', None)
        if total_steps is None and hasattr(self.args, 'num_train_epochs'):
            # Estimate total steps if not provided
            total_steps = getattr(self.args, 'num_train_epochs', 1) * 1000  # rough estimate
            
        warmup_steps = getattr(self.args, 'warmup_steps', 0)
        
        if scheduler_type == "linear":
            return LinearLR(
                self.optimizer, 
                start_factor=1.0, 
                end_factor=0.0, 
                total_iters=total_steps - warmup_steps if total_steps else 1000
            )
        elif scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer, 
                T_max=total_steps - warmup_steps if total_steps else 1000
            )
        elif scheduler_type == "step":
            step_size = getattr(self.args, 'lr_scheduler_step_size', total_steps // 3 if total_steps else 300)
            gamma = getattr(self.args, 'lr_scheduler_gamma', 0.1)
            return StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        
        return None