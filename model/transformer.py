import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):

        batch_size, seq_length, _ = x.size()

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        Q = Q.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1,2)
        out = out.contiguous().view(batch_size, seq_length, self.d_model)
        return self.w_o(out)
    
class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, ff_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.attn = SelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):

        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x
class TinyGPT(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 128,
                 n_heads: int = 8,
                 num_layers: int = 4,
                 ff_hidden: int = 512,
                 max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len,   d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_hidden, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):

        B, T = idx.size()
        tok_emb = self.token_embed(idx)                 
        pos_ids = torch.arange(T, device=idx.device)     
        pos_emb = self.pos_embed(pos_ids).unsqueeze(0)   
        x = self.drop(tok_emb + pos_emb)                 
        
        for block in self.blocks:
            x = block(x)                              

        x = self.ln_f(x)                         
        logits = self.head(x)                         
        return logits




