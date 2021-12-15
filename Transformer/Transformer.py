import torch
import torch.nn as nn
import torchtext

# Query: Target sentence
# Key: Source sentence

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
        keys = values.reshape(N, key_len, self.heads, self.head_dim)
        queries = values.reshape(N, query_len, self.heads, self.head_dim)

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

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        f = self.ff(x)
        out = self.dropout(self.norm2(f + x))
        return out

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, n_layers, heads, device, forward_expansion, dropout, max_len):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)
        self.dropout == nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion,
                )
            ]
        )

    def forward(self, x, mask):
        N, seq_len = x.shape

        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

        self.tansformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )

    def forward(self, x, value, key, source_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.tansformer_block(value, key, query, source_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_size, n_layers, heads, forward_expansion, dropout, device, max_len):
        super(Decoder, self).__init__()

        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(n_layers)]
        )

        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, source_mask, target_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x. encoder_out, encoder_out, source_mask, target_mask)

        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(
        self, 
        source_vocab_size, 
        target_vocabg_size, 
        src_pad_idx, 
        trg_pad_idx, 
        embed_size = 256, 
        n_layers = 6, 
        forward_expansion = 4, 
        heads = 8, 
        dropout = 0, 
        device = 'cuda' if torch.cuda.is_available() else 'cpu', 
        max_len = 100
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            source_vocab_size,
            embed_size,
            n_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_len
        )

        self.decoder = Decoder(
            target_vocabg_size,
            embed_size,
            n_layers,
            heads, 
            forward_expansion,
            dropout,
            device,
            max_len
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_src_mask(trg)

        eng_src = self.encoder(src, src_mask)
        out = self.decoder(trg, eng_src, src_mask, trg_mask)
        return out
