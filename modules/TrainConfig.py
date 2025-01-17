from gc import callbacks
import json
import os
import random
import sys
import torch
from transformers import RobertaTokenizer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer as lTrainer
from lightning.pytorch.loggers import CSVLogger




def init_checkpoint(cpkt_path: str, model_dir: str, version: int, targetMetric: str = 'val_auxilary_loss'):
    return ModelCheckpoint(
        dirpath=cpkt_path,
        filename=f"{model_dir}_v{version}",
        save_top_k=1,
        verbose=True,
        monitor=targetMetric,
        mode='min'
    )

def init_logger(model_dir: str, version: int, log_path: int):
    return CSVLogger(
        save_dir=log_path,
        name=f"{model_dir}_v{version}",
    )
    
def early_stop(targetMetric = 'val_auxillary_input') -> EarlyStopping:
    return EarlyStopping(
        monitor=targetMetric,
        mode='min',
        min_delta=3e-2,
        check_finite=True,
        patience=3,
        strict=True,
    )
    

def Trainer(
        checkpoint: ModelCheckpoint, logger: CSVLogger,
        num_epochs: int, debug=False, precision: str = '32-true',
        targetMetric: str = 'val_auxilary_loss'
    ):
    
    callbacks = [early_stop(),checkpoint]
    return lTrainer(
        callbacks=callbacks,
        max_epochs=num_epochs,
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        fast_dev_run=debug,
        inference_mode=False,
        precision=precision
    )
    
def read_hparams(json_path: str, decoder_start_token_id: int) -> dict:
    if not os.path.exists(json_path):
        raise ValueError("json path does not exist.")
    with open(json_path, 'r') as f:
        hparams = json.load(f)
    hparams['decoder_start_token_id'] = decoder_start_token_id
    return hparams

def masker(code: str, tokenizer: RobertaTokenizer, mask_prob: int = 0.15) -> str:
    tokenized_code = tokenizer.tokenize(code)
    token_ids = tokenizer.convert_tokens_to_ids(tokenized_code)
    masked_token_ids = [token_id if random.random() > mask_prob else tokenizer.mask_token_id for token_id in token_ids]
    masked_tokenized_code = [tokenizer.convert_ids_to_tokens(masked_token_id) for masked_token_id in masked_token_ids]
    
    return tokenizer.convert_tokens_to_string(masked_tokenized_code)