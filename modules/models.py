from typing import Any
from datetime import datetime
from difflib import unified_diff
from lightning.pytorch.utilities.types import STEP_OUTPUT
from mdurl import decode
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import (
    RobertaForMaskedLM,
    T5ForConditionalGeneration,
    RobertaTokenizer
)
import os
import numpy as np
import torch
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns



class CodeBertJS(pl.LightningModule):
    def __init__(
        self, 
        tokenizer: RobertaTokenizer,
        class_weights: np.array, 
        model_dir: str = 'microsoft/codebert-base-mlm', 
        num_classes: int = 6, 
        dropout_rate: float =0.1,
        with_activation: bool = False,
        with_layer_norm: bool = False
        ) -> None:
        super().__init__()
        self.model_dir = model_dir
        self.encoder = RobertaForMaskedLM.from_pretrained(
                    model_dir,
                    output_hidden_states=True,
                    output_attentions=True,
                    num_beams=5,
                    num_beam_groups=2,
                    return_dict=True,
                )
        self.classes = ["mobile","functionality","ui-ux","compatibility-performance","network-security","general"] if num_classes == 6  else ["functionality","ui-ux","compatibility-performance","network-security","general"]
        self.dropout_rate = dropout_rate
        self.with_layer_norm = with_layer_norm
        if self.with_layer_norm:
            self.layer_norm = nn.LayerNorm(self.encoder.config.hidden_size)
        
        self.predictions = []
        self.labels = []
        self.generated_codes = []
        self.with_activation = with_activation
        if self.with_activation:
            hiddenLayerDim = self.encoder.config.hidden_size
            classifierInFeatures = hiddenLayerDim
            self.hidden_layer = nn.Linear(self.encoder.config.hidden_size, hiddenLayerDim)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(p=self.dropout_rate)
            self.classifier = nn.Linear(classifierInFeatures, num_classes)
        else:
            self.dropout = nn.Dropout(p=self.dropout_rate)
            self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)
        self.class_weights = torch.tensor(class_weights)
        self.tokenizer = tokenizer
    
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
        encoder_hidden_states = output.hidden_states[-1]
        encoder_output = apply_layers(encoder_hidden_states)
        
        return encoder_output
    
    def forward(self, batch):
        encoder_output = self.encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['gt_input_ids'],
            output_hidden_states=True
        )
        hidden_states_output = self.classifier_layers(encoder_output)

        classification_logits = self.classifier(hidden_states_output)
        return encoder_output.loss, encoder_output.logits, classification_logits
        

    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        self.encoder.gradient_checkpointing_enable()
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
        self.log("test_loss", classification_loss, prog_bar=True, logger=True)
        return {
            'classification_loss' : classification_loss, 
            'loss': loss,
            'logits': outputs,
            'classification_logits': classification_logits
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
    
    def predict_step(self, input_ids, gt_input_ids):
        return self.model(
            input_ids=input_ids,
        )

    def configure_optimizers(self) -> optim.Adam:
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

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
    
class CodeT5(pl.LightningModule): 
    def __init__(
        self, 
        class_weights: np.array, 
        model_dir: str = 'Salesforce/codet5-base', 
        num_classes: int = 6, 
        dropout_rate=0.1,
        with_activation: bool = False,
        with_layer_norm: bool = False
        ) -> None:
        super().__init__()
        self.model_dir = model_dir
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_dir) 
        self.predictions = []
        self.labels = []
        self.classes = ["mobile","functionality","ui-ux","compatibility-performance","network-security","general"] if num_classes == 6  else ["functionality","ui-ux","compatibility-performance","network-security","general"]
        self.save_hyperparameters()
        self.generated_codes = []
        self.dropout_rate = dropout_rate
        self.with_layer_norm = with_layer_norm
        if self.with_layer_norm:
            self.layer_norm = nn.LayerNorm(self.model.config.d_model)
        
        self.with_activation = with_activation
        if self.with_activation:
            hiddenLayerDim = self.model.config.d_model // 2
            classifierInFeatures = 2 * hiddenLayerDim
            self.hidden_layer = nn.Linear(self.model.config.d_model, hiddenLayerDim)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(p=self.dropout_rate)
            self.classifier = nn.Linear(classifierInFeatures, num_classes)
        else:
            self.dropout = nn.Dropout(p=self.dropout_rate)
            self.classifier = nn.Linear(self.model.config.d_model, num_classes)
            
        self.class_weights = torch.tensor(class_weights)
    
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
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            output_hidden_states=True
        )
        # Get the last hidden state of the encoder output 
        hidden_states_output = self.classifier_layers(output)
        classification_logits = self.classifier(hidden_states_output)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        return optimizer
    
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
