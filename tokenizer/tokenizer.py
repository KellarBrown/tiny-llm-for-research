import re
import json
from collections import Counter
from typing import List, Dict

class Tokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
        self.unk_token = "<unk>"

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()] 
        ids = [self.str_to_int.get(s, self.str_to_int[self.unk_token]) for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids]) 
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
    def build_vocab(cls, text:str, min_freq: int = 1):
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [t.strip() for t in tokens if t.strip()]
    

if __name__ == "__main__":
    vocab = {"Hello": 0, "world": 1, "<unk>": 2}
    tokenizer = Tokenizer(vocab)

    print("Encoded:", tokenizer.encode("Hello you beautiful world"))
    print("Decoded:", tokenizer.decode([0, 2, 2, 1]))
