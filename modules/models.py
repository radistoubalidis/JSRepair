import sys
from transformers import (
    AutoModelForMaskedLM,
    T5ForConditionalGeneration,
    AdamW
    )
import pytorch_lightning as pl
import torch.optim as optim


class CodeBertJS(pl.LightningModule):

    def __init__(self) -> None:
        super().__init__()
        model_path = 'microsoft/codebert-base-mlm'
        # self.save_hyperparameters()
        self.encoder = AutoModelForMaskedLM.from_pretrained(model_path, return_dict=True)
        self.encoder = self.encoder.to(self.device)

    def forward(self, input_ids, attention_mask, labels=None):
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return encoder_output.loss, encoder_output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        loss, outputs = self.forward(input_ids, attention_mask, labels)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        loss, outputs = self.forward(input_ids, attention_mask, labels)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        loss, outputs = self.forward(input_ids, attention_mask, labels)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return optimizer
    
class CodeT5(pl.LightningModule):
    
    def __init__(self) -> None:
        super().__init__()
        model_path = 'Salesforce/codet5-base'
        self.model = T5ForConditionalGeneration.from_pretrained(model_path, return_dict=True)
        
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output.loss , output.logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self.forward(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)


