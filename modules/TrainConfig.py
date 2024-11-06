import json
import os
import random
import sys
import pytorch_lightning as pl
import torch
from transformers import RobertaTokenizer



def init_checkpoint(cpkt_path: str, model_dir: str, version: int):
    return pl.callbacks.ModelCheckpoint(
        dirpath=cpkt_path,
        filename=f"{model_dir}_v{version}",
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

def init_logger(model_dir: str, version: int, log_path: int):
    return pl.loggers.CSVLogger(
        save_dir=log_path,
        name=f"{model_dir}_v{version}",
    )

def Trainer(checkpoint: pl.callbacks.ModelCheckpoint, logger: pl.loggers.CSVLogger, num_epochs: int, debug=False):
    return pl.Trainer(
        callbacks=checkpoint,
        max_epochs=num_epochs,
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        fast_dev_run=debug,
        inference_mode=False
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