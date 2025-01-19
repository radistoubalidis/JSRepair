import sqlite3
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import (
    T5ForConditionalGeneration,
    RobertaTokenizer
)
import os
import numpy as np
import torch
from lightning import LightningModule
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from diff_match_patch import diff_match_patch
from alt_bert import HF_DIR
from modules.filters import add_labels, count_comment_lines, compute_diffs
from sklearn.model_selection import train_test_split
from modules.TrainConfig import Trainer, init_checkpoint, init_logger

HF_DIR = 'Salesforce/codet5-base'

class CodeT5Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, decodings, class_labels):
        self.input_ids = encodings.input_ids
        self.attention_mask = encodings.attention_mask
        self.labels = decodings
        self.class_labels = class_labels

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
            'class_labels': self.class_labels[idx]
        }
        return item

    def __len__(self):
        return len(self.input_ids)

class CodeT5(LightningModule):
    def __init__(
        self,
        class_weights: np.array = None,
        tokenizer = RobertaTokenizer.from_pretrained(HF_DIR),
        model_dir: str = 'Salesforce/codet5-base',
        num_classes: int = 6,
        dropout_rate=0.1,
        with_activation: bool = False,
        with_layer_norm: bool = False,
        lr: float = 1e-3
        ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.lr = lr
        self.model_dir = model_dir
        self.tokenizer = tokenizer
        self.codet5 = T5ForConditionalGeneration.from_pretrained(self.model_dir)
        self.predictions = []
        self.labels = []
        self.classes = ["mobile","functionality","ui-ux","compatibility-performance","network-security","general"] if num_classes == 6  else ["functionality","ui-ux","compatibility-performance","network-security","general"]
        self.generated_codes = []
        self.dropout_rate = dropout_rate
        self.with_layer_norm = with_layer_norm
        if self.with_layer_norm:
            self.layer_norm = nn.LayerNorm(self.codet5.config.d_model)

        self.with_activation = with_activation
        if self.with_activation:
            hiddenLayerDim = self.codet5.config.d_model // 2
            classifierInFeatures = 2 * hiddenLayerDim
            self.hidden_layer = nn.Linear(self.codet5.config.d_model, hiddenLayerDim)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(p=self.dropout_rate)
            self.classifier = nn.Linear(classifierInFeatures, num_classes)
        else:
            self.dropout = nn.Dropout(p=self.dropout_rate)
            self.classifier = nn.Linear(self.codet5.config.d_model, num_classes)

        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights)

    def compute_grad_norm(self, loss, model):
        """
        Compute the gradient norm for a given loss and model.
        """
        self.zero_grad()
        loss.backward(retain_graph=True)
        grad_norm = sum(p.grad.norm(2).item() for p in model.parameters() if p.grad is not None)
        self.zero_grad()
        return grad_norm

    def classifier_layers(self, output):
        def apply_layers(hidden_state):
            # Apply layer normalization
            output = hidden_state

            # Pooling: Average of the sequence length
            output = torch.mean(output, dim=1)
            if self.with_layer_norm:
                output = self.layer_norm(output)

            # Pass it through an activation function using a hidden layer
            if self.with_activation:
                output = self.activation(self.hidden_layer(output))
            # Pass it through a dropout layer before the classifier
            output = self.dropout(output)
            return output

        # Get the last hidden state of the encoder/decoder output
        encoder_hidden_states = output.encoder_last_hidden_state
        decoder_hidden_states = output.decoder_hidden_states[-1]
        encoder_output = apply_layers(encoder_hidden_states)
        decoder_output = apply_layers(decoder_hidden_states)

        # Combine decoder/encoder outputs
        combined_output = torch.cat([encoder_output, decoder_output], dim=-1)
        combined_output = self.dropout(combined_output)
        return combined_output

    def forward(self, batch):
        output = self.codet5(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            output_hidden_states=True,
        )
        # Get the last hidden state of the encoder output
        hidden_states_output = self.classifier_layers(output)
        classification_logits = self.classifier(hidden_states_output)
        return output.loss, output.logits, classification_logits

    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        opt = self.optimizers()
        opt.zero_grad()
        self.codet5.gradient_checkpointing_enable()
        loss, logits, classification_logits = self.forward(batch)
        classification_loss = self.classification_loss(classification_logits, batch['class_labels'])
        t5_grad_norm = self.compute_grad_norm(loss, self.codet5)
        class_grad_norm = self.compute_grad_norm(classification_loss, self.classifier)

        # Dynamic weights
        alpha = class_grad_norm / (t5_grad_norm + class_grad_norm + 1e-8)
        beta = t5_grad_norm / (t5_grad_norm + class_grad_norm + 1e-8)

        auxilary_loss = alpha * loss + beta * classification_loss
        self.manual_backward(auxilary_loss)
        opt.step()
        self.log("t5_loss", loss, prog_bar=True, logger=True)
        self.log("classification_loss", classification_loss, prog_bar=True, logger=True)
        self.log("auxilary_loss", auxilary_loss, prog_bar=True, logger=True)
        return {'t5_logits': logits, 'classification_logits': classification_logits, 'classification_loss' : classification_loss, 'loss': loss, "auxilary_loss" : auxilary_loss}

    def validation_step(self, batch, batch_idx):
        loss, outputs, classification_logits = self.forward(batch)
        classification_loss = self.classification_loss(classification_logits, batch['class_labels'])

        # Static weights for validation
        alpha = 0.65
        beta = 0.35

        val_auxilary_loss = alpha * loss + beta * classification_loss
        self.log("val_t5_loss", loss, prog_bar=True, logger=True)
        self.log("val_classification_loss", classification_loss, prog_bar=True, logger=True)
        self.log("val_auxilary_loss", val_auxilary_loss, prog_bar=True, logger=True)
        return {'val_t5_logits': outputs, 'val_classification_logits': classification_logits, 'val_classification_loss' : classification_loss, 'val_loss': loss, "val_auxilary_loss" : val_auxilary_loss}
        

    def test_step(self, batch, batch_idx):
        loss, outputs, classification_logits = self.forward(batch)
        classification_loss = self.classification_loss(classification_logits, batch['class_labels'])
        
        self.log("t5_loss", loss, prog_bar=True, logger=True)
        self.log("classification_loss", classification_loss, prog_bar=True, logger=True)
        return {
            'logits': outputs, 
            'classification_logits': classification_logits,
            'classification_loss' : classification_loss, 
            'loss': loss
        }

    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int):
        probs = torch.sigmoid(outputs['classification_logits'])
        preds = (probs > 0.5).float()
        generated_code = self.decode_output(outputs['logits'])
        self.generated_codes.append(generated_code)
        self.predictions.append(preds)
        self.labels.append(batch['class_labels'])

    def on_test_epoch_end(self):
        all_predictions = torch.cat(self.predictions).cpu().numpy()
        all_labels = torch.cat(self.labels).cpu().numpy()
        self.conf_matrix_plot(all_labels,all_predictions)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            verbose=True
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def classification_loss(self, logits, labels):
        return nn.functional.binary_cross_entropy_with_logits(
            logits, labels.float(),
            pos_weight=self.class_weights.to(logits.device),
            )

    def conf_matrix_plot(self, all_labels, all_predictions):
        model_name = os.environ['MODEL_NAME']
        version = int(os.environ['VERSION'])
        metrics_path = os.environ['METRICS_PATH']
        if not os.path.exists(f"{metrics_path}/{model_name}_v{version}"):
            os.mkdir(f"{metrics_path}/{model_name}_v{version}")

        num_classes = all_labels.shape[1]
        n_cols = 2
        n_rows = (num_classes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i in range(num_classes):
            cm = confusion_matrix(all_labels[:, i], all_predictions[:, i])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i])
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Ground Truth')
            axes[i].set_title(f'Confusion Matrix for {self.classes[i]}')

        for j in range(num_classes, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(f"{metrics_path}/{model_name}_v{version}/confusion_matrices.png")
        plt.show()

    def decode_output(self, output) -> str:
        tokens = torch.argmax(output, dim=-1)
        code = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        return code

def load_ds(tokenizer: RobertaTokenizer, debug = False, classLabels: dict = {
    "functionality" : 0.,
    "ui-ux" : 0.,
    "compatibility-performance" : 0.,
    "network-security" : 0.,
    "general": 0.
}):
    db_path = 'commitpack-datasets.db' if os.path.exists('commitpack-datasets.db') else '/content/drive/MyDrive/Thesis/commitpack-datasets.db'
    con = sqlite3.connect(db_path)
    ds_df = pd.read_sql_query("select * from commitpackft_classified_train",con)
    if not os.path.exists(db_path):
        raise FileNotFoundError('sqlite3 path doesnt exist.')
    ds_df['old_contents'] = ds_df['old_contents'].apply(lambda code: code.replace('\n', ' '))
    ds_df['new_contents'] = ds_df['new_contents'].apply(lambda code: code.replace('\n', ' '))
    ds_df['class_labels'] = ds_df['bug_type'].apply(lambda bT: add_labels(bT.split(','), classLabels))
    ds_df = ds_df[ds_df['bug_type'] != 'mobile']
    ds_df = ds_df[ds_df['old_contents'].str.len() > 0]
    ds_df['old_contents_comment_lines_count'] = ds_df['old_contents'].apply(lambda sample: count_comment_lines(sample))
    ds_df['new_contents_comment_lines_count'] = ds_df['new_contents'].apply(lambda sample: count_comment_lines(sample))
    # Filter out samples where the sum of comment lines increased more than 3 lines
    # to prevent excessive masking
    ds_df = ds_df[abs(ds_df['old_contents_comment_lines_count'] - ds_df['new_contents_comment_lines_count']) <= 3]
    # Filter out samples with more than 10 comment lines
    ds_df = ds_df[(ds_df['old_contents_comment_lines_count'] < 10) & (ds_df['new_contents_comment_lines_count'] < 10)]
    
    if debug:
        ds_df = ds_df.sample(100)
    
    dmp = diff_match_patch()
    ds_df['num_changes'] = ds_df.apply(lambda sample: compute_diffs(sample, dmp), axis=1)
    # Filter out samples with more than 3 changes in the code
    ds_df = ds_df[ds_df['num_changes'] <= 3]
    
    return ds_df

def combine_pl_nl(codes: pd.DataFrame, new_key, key, tokenizer):
    codes[new_key] = '/* ' + codes['message'] + '*/ ' + tokenizer.sep_token + ' ' + codes[key]
    return codes

def get_dataset(tokenizer, TRAIN_old, TRAIN_new, VAL_old, VAL_new, max_length=512):
    TRAIN_encodings = tokenizer(
        TRAIN_old['input_seq'].tolist(),
        max_length=max_length,
        pad_to_max_length=True,
        return_tensors='pt',
        padding='max_length',
        truncation=True
    )

    VAL_encodings = tokenizer(
        VAL_old['input_seq'].tolist(),
        max_length=max_length,
        pad_to_max_length=True,
        return_tensors='pt',
        padding='max_length',
        truncation=True
    )

    TRAIN_gt = tokenizer(
        TRAIN_new['output_seq'].tolist(),
        max_length=max_length,
        pad_to_max_length=True,
        return_tensors='pt',
        padding='max_length',
        truncation=True
    ).input_ids

    VAL_gt = tokenizer(
        VAL_new['output_seq'].tolist(),
        max_length=max_length,
        pad_to_max_length=True,
        return_tensors='pt',
        padding='max_length',
        truncation=True
    ).input_ids
    
    TRAIN_classes = torch.tensor(TRAIN_old['class_labels'].tolist())
    VAL_classes = torch.tensor(VAL_old['class_labels'].tolist())
    
    TRAIN_dataset = CodeT5Dataset(encodings=TRAIN_encodings, class_labels=TRAIN_classes, decodings=TRAIN_gt)
    VAL_dataset = CodeT5Dataset(encodings=VAL_encodings, class_labels=VAL_classes, decodings=VAL_gt)
    
    # Class weights
    # pos_weight[i] = (Number of negative samples for class i) / (Number of positive samples for class i)
    num_samples = TRAIN_classes.size(0)
    num_classes = TRAIN_classes.size(1)

    pos_counts = torch.sum(TRAIN_classes, dim=0)
    neg_counts = num_samples - pos_counts
    class_weights = neg_counts / (pos_counts + 1e-6)
    class_weights = class_weights.numpy()
    
    return {
        'TRAIN_encodings': TRAIN_encodings,
        'VAL_encodings': VAL_encodings,
        'TRAIN_gt': TRAIN_gt,
        'VAL_gt': VAL_gt,
        'TRAIN_classes': TRAIN_classes,
        'VAL_classes': VAL_classes,
        'TRAIN_dataset': TRAIN_dataset,
        'VAL_dataset': VAL_dataset,
        'num_classes': num_classes,
        'class_weights': class_weights,
    }
    

def main():
    debug = True if int(input('Debug Run (0,1): ')) == 1 else False
    tokenizer = RobertaTokenizer.from_pretrained(HF_DIR)
    ds_df = load_ds(tokenizer, debug=debug)
    old_codes = ds_df[['message', 'old_contents', 'class_labels']]
    new_codes = ds_df[['message', 'new_contents', 'class_labels']]
    
    bimodal_train = True if input('Combine commit messages with codes (0,1): ') == 1 else False
    if bimodal_train:
        old_codes = combine_pl_nl(old_codes, 'input_seq', 'old_contents', tokenizer)
        new_codes = combine_pl_nl(new_codes, 'output_seq', 'new_contents', tokenizer)
    else:
        old_codes['input_seq'] = old_codes['old_contents'].copy()
        new_codes['output_seq'] = new_codes['new_contents'].copy()
    
    TRAIN_old, VAL_old, TRAIN_new, VAL_new = train_test_split(old_codes, new_codes, test_size=0.3, random_state=42)

    print(f"Total training samples: {len(TRAIN_old)}")
    print(f"Total validation samples: {len(VAL_old)}")
    try:
        TOKENIZER_MAX_LENGTH = int(input('Type tokenizer max length (256, 512, ...): '))
    except:
        TOKENIZER_MAX_LENGTH = 512
        
    LOAD_FROM_CPKT = input("Load from existing model (type cpkt path if true): ")
    NEW_CKPT = False if int(input('Create new checkpoint file (0,1): ')) == 0 else True
    params = get_dataset(
        tokenizer=tokenizer,
        TRAIN_old=TRAIN_old,
        TRAIN_new=TRAIN_new,
        VAL_old=VAL_old,
        VAL_new=VAL_new,
        max_length=TOKENIZER_MAX_LENGTH
    )
    try:
        LEARNING_RATE = float(input('Type initial learning rate (float: 1e-3): '))
    except:
        LEARNING_RATE = 1e-3
        
    DROPOUT_RATE = float(input('Type dropout rate for classifier: '))
    if len(LOAD_FROM_CPKT) > 0 and  os.path.exists(LOAD_FROM_CPKT):
        model = CodeT5.load_from_checkpoint(
            LOAD_FROM_CPKT,
            class_weights=params['class_weights'],
            num_classes=params['num_classes'],
            dropout_rate=params['DROPOUT_RATE'],
            with_activation=True,
            with_layer_norm=True,
            tokenizer=tokenizer,
            lr=LEARNING_RATE
        )
    else:
        model = CodeT5(
            class_weights=params['class_weights'],
            num_classes=params['num_classes'],
            dropout_rate=DROPOUT_RATE,
            with_activation=True,
            with_layer_norm=True,
            tokenizer=tokenizer,
            lr=LEARNING_RATE
        )
        model.codet5.train()
        model.classifier.train()
        
    if debug:
        model.to('cpu')
    
    BATCH_SIZE = 4 if debug else 64
    
    dataloader = DataLoader(params['TRAIN_dataset'], batch_size=BATCH_SIZE,num_workers=14, shuffle=True)
    val_dataloader = DataLoader(params['VAL_dataset'], batch_size=BATCH_SIZE, num_workers=14)
    
    
    modelSize = HF_DIR.split('/')[-1]
    MODEL_DIR = f"CodeBert_{modelSize}_JS_{params['num_classes']}_classes_{TOKENIZER_MAX_LENGTH}MaxL"
    VERSION = int(input('Type train version: '))
    LOG_PATH = 'logs' if os.path.exists('logs') else '/content/drive/MyDrive/Thesis/logs'
    CPKT_PATH = 'checkpoints' if os.path.exists('checkpoints') else '/content/drive/MyDrive/Thesis/checkpoints'

    logger = init_logger(log_path=LOG_PATH, model_dir=MODEL_DIR, version=VERSION)
    checkpoint = init_checkpoint(cpkt_path=CPKT_PATH, model_dir=MODEL_DIR, version=VERSION, targetMetric='val_auxilary_loss')
    
    NUM_EPOCHS = int(input('Type number of epochs: '))
    
    if debug:
        trainer = Trainer(checkpoint=checkpoint,logger=logger,debug=debug, num_epochs=NUM_EPOCHS)
    else:
        trainer = Trainer(checkpoint=checkpoint,logger=logger,debug=debug, num_epochs=NUM_EPOCHS, precision='32-true')

    print('Starting training script..')
    if len(LOAD_FROM_CPKT) > 0 and os.path.exists(LOAD_FROM_CPKT) and not NEW_CKPT:
        trainer.fit(
            model,
            train_dataloaders=dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=LOAD_FROM_CPKT
        )
    else:
        trainer.fit(
            model,
            train_dataloaders=dataloader,
            val_dataloaders=val_dataloader
        )
        