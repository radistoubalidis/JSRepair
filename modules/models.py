import json
import sys
from regex import B
from torch import is_tensor, tensor
import torch
from transformers import (
    RobertaForMaskedLM,
    T5ForConditionalGeneration,
    T5Config,
    RobertaTokenizer
    )
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from difflib import unified_diff

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
    def __init__(self, model_dir: str = 'Salesforce/codet5-base', num_classes: int = 6, ) -> None:
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.classifier = nn.Linear(self.model.config.d_model, num_classes)
        self.save_hyperparameters()
        
    def forward(self, batch):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        classification_logits = self.classifier(output.logits[:, -1, :])
        print(classification_logits)
        raise ValueError
        return output.loss, output.logits, classification_logits
    
    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
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
        return {'classification_loss' : classification_loss, 'loss': loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0001)
    
    def classification_loss(self, logits, labels):
        return nn.functional.binary_cross_entropy_with_logits(logits, labels.float())    
    
    
class T5JSRephraser(pl.LightningModule):
    def __init__(self, t5config: T5Config) -> None:
        super().__init__()
        self.model = T5ForConditionalGeneration(t5config)
        self.save_hyperparameters()
        
    def forward(self, batch):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        return output.loss, output.logits
    
    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        loss, outputs = self.forward(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, outputs = self.forward(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, outputs = self.forward(batch)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, num_beams=3):
        return self.model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
            num_beams=num_beams
        )
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001)