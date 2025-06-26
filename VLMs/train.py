from lightning_module import LLaVALitModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Tạo model và tokenizer như cũ
model = create_llava_model(...)
tokenizer = create_tokenizer(...)

# LightningModule
lit_model = LLaVALitModule(model=model, training_args=training_args, tokenizer=tokenizer)

# Logger + Checkpoint
logger = TensorBoardLogger("logs", name="llava")
checkpoint = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")

# Trainer
trainer = Trainer(
    max_epochs=training_args.num_train_epochs,
    accelerator="gpu",
    devices=1,
    accumulate_grad_batches=training_args.gradient_accumulation_steps,
    precision=training_args.precision,
    logger=logger,
    callbacks=[checkpoint],
)

# Fit
trainer.fit(lit_model, datamodule=your_lightning_datamodule)
