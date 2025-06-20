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