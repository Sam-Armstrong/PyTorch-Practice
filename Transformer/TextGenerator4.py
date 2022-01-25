import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

import math

def make_mask():
    return 0

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p = dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, heads, emsize, d_hid):
        super(TransformerBlock, self).__init__()

        self.attention = nn.MultiheadAttention(emsize, heads, dropout = 0) # Need to mask
        self.fc1 = nn.Linear(emsize, d_hid)
        self.fc2 = nn.Linear(d_hid, emsize)
        self.relu = nn.ReLU()
        self.ln1 = nn.LayerNorm(emsize)
        self.ln2 = nn.LayerNorm(emsize)

    def forward(self, x):
        # x shape: (batch_size, emsize, emsize (for the mask))
        res = x.clone()
        x = self.attention(x, x, x, attn_mask = None)
        x += res
        x = self.ln1(x)

        res = x.clone()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x += res
        x = self.ln2(x)

        return x


if __name__ == '__main__':
    block = TransformerBlock(1, 100, 100)