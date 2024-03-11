import sys
import torch

class CodeSearchNetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, sample_id):
        # Permutate the tokenized input_ids to feed as "bad sample"
        input_ids_copy = encodings.copy().input_ids
        self.input_ids = input_ids_copy[:, torch.randperm(input_ids_copy.size(1))]
        self.attention_mask = encodings.attention_mask
        # Labels are the "good_samples"
        self.labels = encodings.input_ids

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.input_ids)


class CommitPackDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.input_ids = encodings.input_ids
        self.attention_mask = encodings.attention_mask
        self.labels = labels.input_ids
        
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
        }
    
    def __len__(self):
        return len(self.input_ids)
    
class CodeT5Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, decodings):
        self.input_ids = encodings.input_ids
        self.attention_mask = encodings.attention_mask
        self.decoder_input_ids = decodings.input_ids
        self.decoder_attention_mask = decodings.attention_mask

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'decoder_input_ids': self.decoder_input_ids[idx],
            'decoder_attention_mask': self.decoder_attention_mask[idx]
        }
        return item

    def __len__(self):
        return len(self.input_ids)