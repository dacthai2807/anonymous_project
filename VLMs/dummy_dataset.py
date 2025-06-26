import torch
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, length=100, seq_len=128, vocab_size=1000):
        self.length = length
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len),
            "labels": torch.randint(0, self.vocab_size, (self.seq_len,)),
            "images": torch.randn(3, 224, 224)
        }
