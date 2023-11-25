import pytorch_lightning as pl
import torch

MAX_EPOCHS = 10

def init_checkpoint(cpkt_path, model_dir, version):
    return pl.callbacks.ModelCheckpoint(
        dirpath=cpkt_path,
        filename=f"{model_dir}_v{version}",
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

def init_logger(log_path, model_dir, version):
    return pl.loggers.CSVLogger(
        save_dir=log_path,
        name=f"{model_dir}_v{version}",
    )

def Trainer(checkpoint: pl.callbacks.ModelCheckpoint, logger: pl.loggers.CSVLogger, debug=False):
    return pl.Trainer(
        callbacks=checkpoint,
        max_epochs=MAX_EPOCHS,
        logger=logger,
        devices=1,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        fast_dev_run=debug,
    )