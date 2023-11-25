import argparse
from http.client import HTTPException
import json
from re import DEBUG
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


tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base-mlm')
MAX_SAMPLE = 100000

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
    rows = [x['row'] for x in response_json['rows']]
    dataset_list = [{'buggy_code': x['old_contents'], 'fixed_code':x['new_contents']} for x in rows]
    df = pd.DataFrame(dataset_list)
    train_inputs , train_labels = df['buggy_code'].tolist(), df['fixed_code'].tolist()
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
    return parser.parse_args()

def train(X_train, X_val, Y_train, Y_val, debug_mode: bool):
    train_dataset = CommitPackDataset(X_train, Y_train)
    val_dataset = CommitPackDataset(X_val, Y_val)
    train_dataloder = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1)
    print("Dataset Ready.")
    model = CodeBertJS()
    logger = init_logger('log_test','CodeBertJS2', 1)
    checkpoint = init_checkpoint('checkpoint_test', 'CodeBertJS2', 1)
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
        train(X_train, X_val, Y_train, Y_val, DEBUG)
        offset += 100
        if offset >= MAX_SAMPLE:
            break

if __name__ == '__main__':
    main()
        
        