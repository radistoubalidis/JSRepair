import sys
from torch import tensor
from transformers import (
    AutoModelForMaskedLM,
    RobertaForMaskedLM,
    T5ForConditionalGeneration,
    AdamW,
    RobertaTokenizer,
    DataCollatorForLanguageModeling,
    )
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn

class CodeBertJS(pl.LightningModule):

    def __init__(self) -> None:
        super().__init__()
        model_path = 'microsoft/codebert-base-mlm'
        # self.save_hyperparameters()
        self.encoder = RobertaForMaskedLM.from_pretrained(model_path, return_dict=True)
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
        self.model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base', return_dict=True)
        
    def forward(self, batch):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            decoder_input_ids=batch['decoder_input_ids'],
            decoder_attention_mask=batch['decoder_attention_mask']
        )
        sys.exit(dir(output.logits))
        return output
    
    def training_step(self, batch, batch_idx):
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
    
    def configure_optimizers(self) -> AdamW:
        return AdamW(self.parameters(), lr=0.0001)