import json
import os
import sys
import pytorch_lightning as pl
import torch

MAX_EPOCHS = 3

def init_checkpoint(cpkt_path, model_dir, version):
    return pl.callbacks.ModelCheckpoint(
        dirpath=cpkt_path,
        filename=f"{model_dir}_v{version}",
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

def init_logger(model_dir, version, log_path='/home/disras/projects/JSRepair/logs'):
    return pl.loggers.CSVLogger(
        save_dir=log_path,
        name=f"{model_dir}_v{version}",
    )

def Trainer(checkpoint: pl.callbacks.ModelCheckpoint, logger: pl.loggers.CSVLogger, debug=False):
    return pl.Trainer(
        callbacks=checkpoint,
        max_epochs=MAX_EPOCHS,
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        fast_dev_run=debug,
        inference_mode=False
    )
    
def read_hparams(json_path: str) -> dict:
    if not os.path.exists(json_path):
        raise ValueError("json path does not exist.")
    with open(json_path, 'r') as f:
        hparams = json.load(f)
    return hparams