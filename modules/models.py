import json
import sys
from regex import B
from torch import is_tensor, tensor
import torch
from transformers import (
    RobertaForMaskedLM,
    T5ForConditionalGeneration,
    T5Config,
    )
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn

class CodeBertJS(pl.LightningModule):
    def __init__(self) -> None:
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
    
    def forward(self, input_ids, attention_mask, gt_input_ids):
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Cross Entropy loss between output logits and gt_input_ids
        loss = self.criterion(encoder_output.logits.view(-1, self.encoder.config.vocab_size), gt_input_ids.view(-1))
        
        return loss, encoder_output.logits

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
        loss, outputs = self.forward(input_ids, attention_mask, gt_input_ids)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        gt_input_ids = batch['gt_input_ids']
        loss, outputs = self.forward(input_ids, attention_mask, gt_input_ids)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    
class CodeT5(pl.LightningModule): 
    def __init__(self, mode: str = 'train') -> None:
        super().__init__()
        self.mode = mode
        self.model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
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
        return torch.optim.AdamW(self.parameters(), lr=0.0001)
    
    
    
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