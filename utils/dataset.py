import torch
from torch.utils.data import Dataset
from tokenizer.tokenizer import Tokenizer

class TextDataset(Dataset):

    def __init__(self, file_path: str, seq_length: int = 128, min_freq: int = 1):

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        self.tokenizer = Tokenizer.build_vocab(text, min_freq=min_freq)    
        self.token_ids = self.tokenizer.encode(text)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.token_ids) - self.seq_length


    def __getitem__(self, idx):
        x = self.token_ids[idx : idx + self.seq_length]
        y = self.token_ids[idx + 1 : idx + self.seq_length + 1]
        return torch.tensor(x), torch.tensor(y)