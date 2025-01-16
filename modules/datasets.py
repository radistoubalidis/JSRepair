import torch

class CodeBertDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, gt_input_ids, class_labels):
        self.input_ids = encodings.input_ids
        self.attention_mask = encodings.attention_mask
        self.class_labels = class_labels
        self.gt_input_ids = gt_input_ids

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'class_labels': self.class_labels[idx],
            'gt_input_ids': self.gt_input_ids[idx]
        }
        return item

    def __len__(self):
        return len(self.input_ids)
    
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
            'labels': self.labels.input_ids[idx],
            'class_labels': self.class_labels[idx]
        }
        return item

    def __len__(self):
        return len(self.input_ids)