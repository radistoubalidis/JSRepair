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
from lightning import LightningModule
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns



class CodeBertJS(LightningModule):
    def __init__(
        self, 
        tokenizer: RobertaTokenizer,
        class_weights: np.array, 
        model_dir: str = 'microsoft/codebert-base-mlm', 
        num_classes: int = 6, 
        dropout_rate: float =0.1,
        with_activation: bool = False,
        with_layer_norm: bool = False,
        lr: float = 1e-3
        ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.model_dir = model_dir
        self.lr = lr
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
        self.codebert_loss = nn.CrossEntropyLoss()
        
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
        encoder_hidden_states = output.hidden_states[-1]
        encoder_output = apply_layers(encoder_hidden_states)
        
        return encoder_output
    
    def forward(self, batch):
        encoder_output = self.encoder(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=True
        )
        encoder_loss = self.codebert_loss(encoder_output.logits.view(-1, encoder_output.logits.size(-1)),batch['gt_input_ids'].view(-1))

        hidden_states_output = self.classifier_layers(encoder_output)

        classification_logits = self.classifier(hidden_states_output)
        return encoder_loss, encoder_output.logits, classification_logits

    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        self.encoder.gradient_checkpointing_enable()
        opt = self.optimizers()
        opt.zero_grad()
        self.encoder.gradient_checkpointing_enable()
        loss, outputs, classification_logits = self.forward(batch)
        classification_loss = self.classification_loss(classification_logits, batch['class_labels'])
        bert_grand_norm = self.compute_grad_norm(loss, self.encoder)
        class_grand_norm = self.compute_grad_norm(classification_loss, self.classifier)
        
        # Dynamic Loss Weights
        alpha = class_grand_norm / (bert_grand_norm + class_grand_norm + 1e-8)
        beta = bert_grand_norm / (bert_grand_norm + class_grand_norm + 1e-8)
        
        auxilary_loss = alpha * loss + beta * classification_loss
        self.manual_backward(auxilary_loss)
        opt.step()
        
        self.log("bert_loss", loss, prog_bar=True, logger=True)
        self.log("classification_loss", classification_loss, prog_bar=True, logger=True)
        self.log("auxilary_loss", auxilary_loss, prog_bar=True, logger=True)
        return {'classification_loss' : classification_loss, 'loss': loss, "auxilary_loss" : auxilary_loss}

    def validation_step(self, batch, batch_idx):
        loss, outputs, classification_logits = self.forward(batch)
        classification_loss = self.classification_loss(classification_logits, batch['class_labels'])
        
        # Static weights for validation
        alpha = 0.65
        beta = 0.35
        
        val_auxilary_loss = alpha * loss + beta * classification_loss
        
        self.log("val_bert_loss", loss, prog_bar=True, logger=True)
        self.log("val_classification_loss", classification_loss, prog_bar=True, logger=True)
        self.log("val_auxilary_loss", val_auxilary_loss, prog_bar=True, logger=True)
        return {'val_classification_loss' : classification_loss, 'val_loss': loss, "val_auxilary_loss" : val_auxilary_loss}

    def test_step(self, batch, batch_idx):
        loss, outputs, classification_logits = self.forward(batch)
        classification_loss = self.classification_loss(classification_logits, batch['class_labels'])
        
        # Static weights for validation
        alpha = 0.65
        beta = 0.35
        
        auxilary_loss = alpha * loss + beta * classification_loss
        self.log("bert_loss", loss, prog_bar=True, logger=True)
        self.log("classification_loss", classification_loss, prog_bar=True, logger=True)
        self.log("auxilary_loss", auxilary_loss, prog_bar=True, logger=True)
        return {
            'auxilary_loss': auxilary_loss,
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
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            verbose=True
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

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
    
class CodeT5(LightningModule):
    def __init__(
        self,
        class_weights: np.array = None,
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
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_dir)
        self.predictions = []
        self.labels = []
        self.classes = ["mobile","functionality","ui-ux","compatibility-performance","network-security","general"] if num_classes == 6  else ["functionality","ui-ux","compatibility-performance","network-security","general"]
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
        output = self.model(
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
        self.model.gradient_checkpointing_enable()
        loss, outputs, classification_logits = self.forward(batch)
        classification_loss = self.classification_loss(classification_logits, batch['class_labels'])
        t5_grad_norm = self.compute_grad_norm(loss, self.model)
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
        return {'classification_loss' : classification_loss, 'loss': loss, "auxilary_loss" : auxilary_loss}

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
        return {'val_classification_loss' : classification_loss, 'val_loss': loss, "val_auxilary_loss" : val_auxilary_loss}

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
        self.generated_codes.append(self.decode_output(outputs['logits']))
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
