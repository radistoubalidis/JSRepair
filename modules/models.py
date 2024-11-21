import os
from typing import Any
import numpy as np
from regex import B
from torch import Value, is_tensor, tensor
from datetime import datetime
from difflib import unified_diff
from lightning.pytorch.utilities.types import STEP_OUTPUT
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from transformers import (
    RobertaForMaskedLM,
    T5ForConditionalGeneration,
    RobertaTokenizer,
    get_scheduler
    )
import torch
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns



class CodeBertJS(pl.LightningModule):
    def __init__(self, tokenizer: RobertaTokenizer) -> None:
        super().__init__()
        self.encoder = RobertaForMaskedLM.from_pretrained(
                    'microsoft/codebert-base-mlm',
                    output_hidden_states=True,
                    output_attentions=True,
                    num_beams=5,
                    num_beam_groups=2,
                    return_dict=True,
                )
        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.last_validation_batch = None
        self.last_validation_output = None
    
    def forward(self, input_ids, attention_mask = None, gt_input_ids = None):
        if attention_mask is not None and gt_input_ids is not None:
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Cross Entropy loss between output logits and gt_input_ids
            loss = self.criterion(encoder_output.logits.view(-1, self.encoder.config.vocab_size), gt_input_ids.view(-1))
            return loss, encoder_output.logits
        else:
            return self.encoder(input_ids)
        

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        gt_input_ids = batch['gt_input_ids']
        loss, outputs = self.forward(input_ids, attention_mask, gt_input_ids)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        gt_input_ids = batch['gt_input_ids']
        self.last_validation_batch = batch
        loss, outputs = self.forward(input_ids, attention_mask, gt_input_ids)
        self.last_validation_output = outputs
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        gt_input_ids = batch['gt_input_ids']
        loss, outputs = self.forward(input_ids, attention_mask, gt_input_ids)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, input_ids, gt_input_ids):
        return self.model(
            input_ids=input_ids,
        )
        
    def on_validation_end(self):
        last_input_tokens = self.tokenizer.convert_ids_to_tokens(self.last_validation_batch['input_ids'][-1])
        last_gt_tokens = self.tokenizer.convert_ids_to_tokens(self.last_validation_batch['gt_input_ids'][-1])
        last_output_tokens = torch.argmax(self.last_validation_output, dim=-1).tolist()
        last_input_seq = self.tokenizer.convert_tokens_to_string(last_input_tokens).replace('<s>','').replace('</s>','').replace('<pad>','')
        last_gt_seq = self.tokenizer.convert_tokens_to_string(last_gt_tokens).replace('<s>','').replace('</s>','').replace('<pad>','')
        codeDiff = unified_diff(last_input_seq, last_gt_seq)
        codeDiffStr = "\n".join(codeDiff)
        last_output_seq = self.tokenizer.batch_decode(last_output_tokens[-1], skip_special_tokens=True)
        print(last_output_seq)
        raise ValueError
        last_output_seq = self.tokenizer.convert_tokens_to_string(last_output_seq)
        
        log_msg = f"""
        Val Batch Sample:
        --------------------Output Code-------------------
        {last_output_seq}
        --------------------Code Diff---------------------
        {codeDiffStr}
        """
        with open(f"train-{datetime.today().strftime('%Y-%m-%d')}-CodeBert.log", 'a') as f:
            f.write(log_msg)

    def configure_optimizers(self) -> optim.Adam:
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    
class CodeT5(pl.LightningModule): 
    def __init__(self, class_weights: np.array, model_dir: str = 'Salesforce/codet5-base', num_classes: int = 6, dropout_rate=0.1, ) -> None:
        super().__init__()
        self.model_dir = model_dir
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.classifier = nn.Linear(self.model.config.d_model, num_classes)
        self.predictions = []
        self.labels = []
        self.classes = ["mobile","functionality","ui-ux","compatibility-performance","network-security","general"]
        self.save_hyperparameters()
        self.confusion_matrices = []
        self.generated_codes = []
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.hidden_layer = nn.Linear(self.model.config.d_model, 256)
        self.activation = nn.ReLU()
        self.class_weights = class_weights
        
    def forward(self, batch):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        # 1.Get the last hidden state of the output 
        encoder_hidden_states = output.encoder_last_hidden_state
        # 2. Normilize it
        pooled_output = torch.mean(encoder_hidden_states, dim=1)
        # 3. Pass it through a dropout layer before the classifier
        pooled_output = self.dropout(pooled_output)
        # 4. Pass it through an activation function using a hidden layer
        hidden_output = self.activation(self.hidden_layer(pooled_output))
        
        classification_logits = self.classifier(hidden_output)
        return output.loss, output.logits, classification_logits
    
    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        self.model.gradient_checkpointing_enable()
        loss, outputs, classification_logits = self.forward(batch)
        classification_loss = self.classification_loss(classification_logits, batch['class_labels'])
        
        self.log("train_loss", classification_loss, prog_bar=True, logger=True)
        return {'classification_loss' : classification_loss, 'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        loss, outputs, classification_logits = self.forward(batch)
        classification_loss = self.classification_loss(classification_logits, batch['class_labels'])
        self.log("val_loss", classification_loss, prog_bar=True, logger=True)
        return {'classification_loss' : classification_loss, 'loss': loss}
    
    def test_step(self, batch, batch_idx):
        loss, outputs, classification_logits = self.forward(batch)
        classification_loss = self.classification_loss(classification_logits, batch['class_labels'])
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return {
            'logits': outputs, 
            'classification_logits': classification_logits,
            'classification_loss' : classification_loss, 
            'loss': loss
        }
        
    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int):
        probs = torch.sigmoid(outputs['classification_logits'])
        preds = (probs > 0.5).float()
        self.generated_codes.append(self.decode_output(outputs['logits']))
        self.predictions.append(preds)
        self.labels.append(batch['class_labels'])
    
    def on_test_epoch_end(self):
        all_predictions = torch.cat(self.predictions).cpu().numpy()
        all_labels = torch.cat(self.labels).cpu().numpy()
        self.conf_matrix_plot(all_labels,all_predictions)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        scheduler = get_scheduler('linear', optimizer, num_warmup_steps=500, num_training_steps=3)
        return [optimizer], [scheduler]
    
    def classification_loss(self, logits, labels):
        return nn.functional.binary_cross_entropy_with_logits(
            logits, labels.float(), pos_weight=self.class_weights.to(logits.device)
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