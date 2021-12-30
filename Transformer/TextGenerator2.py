## Create transformer model from scratch for text generation
# Mask both the input and target tensors for training

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from Transformer.Transformer import DecoderBlock

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), 'Embed size needs to be divisible by heads'

        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])
        # queries: (N, query_len, heads, head_dim)
        # keys: (N, key_len, heads, head_dim)
        # energy: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3)

        out = torch.einsum('nhql,nlhd->nqhd', [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        # attention: (N, heads, query_len, key_len)
        # values: (N, value_len, heads, heads_dim)
        # out: (N, query_len, heads, head_dim) then flatten last two dimensions

        out = self.fc_out(out)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_hid, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') , dropout = 0) -> None:
        super(EncoderLayer, self).__init__()
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(embed_dim, d_hid)
        self.fc2 = nn.Linear(d_hid, embed_dim)
        self.attention = SelfAttention(embed_dim, num_heads)
        # self.ln1 = nn.LayerNorm() # Normalized shape?
        # self.ln2 = nn.LayerNorm()

    def forward(self, x, mask):
        res = x.clone()
        x = self.attention(x, x, x, mask)
        x += res
        #x = self.ln1(x)

        res = x.clone()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x += res
        #x = self.ln2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_hid) -> None:
        super(DecoderLayer, self).__init__()
        
        self.attention1 = SelfAttention(embed_dim, num_heads)
        self.attention2 = SelfAttention(embed_dim, num_heads)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(embed_dim, d_hid)
        self.fc2 = nn.Linear(d_hid, embed_dim)
        # self.ln1 = nn.LayerNorm()
        # self.ln2 = nn.LayerNorm()
        # self.ln3 = nn.LayerNorm()

    def forward(self, x, enc_out, mask):
        res = x.clone()
        x = self.attention1(x, x, x, mask)
        x += res
        #x = self.ln1(x)

        res = x.clone()
        x = self.attention2(enc_out, enc_out, x, None)
        x += res
        #x = self.ln2(x)

        res = x.clone()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x += res
        #x = self.ln3(x)

        return x

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float(0), diagonal=1)

class Generator(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, d_hid, n_layers = 1, dropout = 0, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> None:
        super(Generator, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_hid = d_hid
        self.dropout = dropout

        self.enc_embedding = nn.Embedding(vocab_size, embed_dim)
        self.dec_embedding = nn.Embedding(vocab_size, embed_dim)

        self.enc_layers = nn.ModuleList(
            [EncoderLayer(embed_dim, num_heads, d_hid) for _ in range(n_layers)]
        )

        self.dec_layers = nn.ModuleList(
            [DecoderLayer(embed_dim, num_heads, d_hid) for _ in range(n_layers)]
        )

        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # input shape: (batch_size, seq_len)
        print(x.shape)
        enc_out = self.enc_embedding(x)
        print(x.shape)
        #src_mask = 
        for layer in self.enc_layers:
            enc_out = layer(enc_out)

        return x


if __name__ == '__main__':
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    target = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    g = Generator()
    g.forward()