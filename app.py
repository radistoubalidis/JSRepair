import argparse
from http.client import HTTPException
import json
import os
from re import DEBUG
import sys
from modules.clients import HuggingFaceClient
from modules.datasets import CommitPackDataset
from modules.TrainConfig import Trainer, init_checkpoint, init_logger
from transformers import (
    RobertaTokenizer
)
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from modules.models import CodeBertJS
from modules.miner import get_fix_commits


tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base-mlm')
model = CodeBertJS()

MAX_SAMPLE = 100000

if os.path.exists('/content/drive/MyDrive/Thesis/checkpoints'):
    CPKT_PATH = '/content/drive/MyDrive/Thesis/checkpoints'
else:
    CPKT_PATH = 'checkpoint_test'

if os.path.exists('/content/drive/MyDrive/Thesis/logs'):
    LOG_PATH = '/content/drive/MyDrive/Thesis/logs'
else:
    LOG_PATH = 'log_test'

def download_split(client: HuggingFaceClient,offset: int):
    method = 'rows'
    config = {
        'dataset' : "bigcode/commitpack",
        'config' : 'javascript',
        'split' : 'train',
        'offset' : offset,
        'length' : 100,
    }

    response = client.get(method, config)
    
    if response.status_code != 200:
        print(f"Get Error from Hugging Face client.\n Trained on {offset} samples.\nStoping Training Script.")
        raise HTTPException({response.json()})
    
    response_json = response.json()
    dataset_df = get_fix_commits(response_json)
    print(f"""
            Dataset Samples
            {dataset_df.head()}
        """)
    train_inputs , train_labels = dataset_df['old_contents'].tolist(), dataset_df['new_contents'].tolist()
    return train_test_split(train_inputs, train_labels, test_size=0.1, random_state=42)
    

def tokenize(inputs):
    return tokenizer(
        inputs,
        max_length=508,
        pad_to_max_length=True,
        return_tensors='pt',
        padding='max_length',
        truncation=True
    )

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offset')
    parser.add_argument('--debug')
    parser.add_argument('--version')
    return parser.parse_args()

def train(X_train, X_val, Y_train, Y_val, debug_mode: bool, version: int):
    train_dataset = CommitPackDataset(X_train, Y_train)
    val_dataset = CommitPackDataset(X_val, Y_val)
    train_dataloder = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    print("Dataset Ready.")
    logger = init_logger(LOG_PATH,'CodeBertJS2', version)
    checkpoint = init_checkpoint(CPKT_PATH, 'CodeBertJS2', version)
    trainer = Trainer(checkpoint,logger,debug=debug_mode)
    trainer.fit(
        model,
        train_dataloaders=train_dataloder,
        val_dataloaders=val_dataloader
    )
    
    
def main():
    args = parseArgs()
    INITIAL_OFFSET = args.offset
    DEBUG = True if args.debug == 1 else False
    VERSION = args.version
    offset = int(INITIAL_OFFSET)
    while(True):
        print(f"{offset // 100}-th hundred.")
        if offset == INITIAL_OFFSET:
            X_train, X_val, Y_train, Y_val = download_split(HuggingFaceClient(), INITIAL_OFFSET)
        else:
            X_train, X_val, Y_train, Y_val = download_split(HuggingFaceClient(), offset)
        X_train = tokenize(X_train)
        X_val = tokenize(X_val)
        Y_train = tokenize(Y_train)
        Y_val = tokenize(Y_val)
        train(X_train, X_val, Y_train, Y_val, DEBUG, VERSION)
        offset += 100
        if offset >= MAX_SAMPLE:
            break   

if __name__ == '__main__':
    main()
        
        