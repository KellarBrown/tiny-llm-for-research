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
    @classmethod
    def build_vocab(cls,text:str, min_freq: int = 1):
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [t.strip() for t in tokens if t.strip()]
        counter = Counter(tokens)
        filtered_tokens = [token for token, freq in counter.items() if freq >= min_freq]
        if "<unk>" not in filtered_tokens:
            filtered_tokens.append("<unk>")
        vocab = {token: idx for idx, token in enumerate(filtered_tokens)}
        return cls(vocab)
    
if __name__ == "__main__":

    sample_text = "Friend, Romans, countrymen, lend me your ears. dlkfjdk"
    tokenizer = Tokenizer.build_vocab(sample_text, min_freq=2)
    print("Vocab:", tokenizer.str_to_int)
    print("Encode:", tokenizer.encode("Friend, Romans, countrymen, lend me your ears. dlkfjdk"))