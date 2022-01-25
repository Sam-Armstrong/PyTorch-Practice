## Create transformer model from scratch for text generation
# Mask both the input and target tensors for training

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import dataset
import pytorch_forecasting
import time
import math
from typing import Tuple
import copy

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
            try:
                energy = energy.masked_fill(mask == 0, float('-1e20'))
            except:
                print('ERROR')
                print('Energy shape: ', energy.shape)
                print('Mask shape: ', mask.shape)

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3)

        out = torch.einsum('nhql,nlhd->nqhd', [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        # attention: (N, heads, query_len, key_len)
        # values: (N, value_len, heads, heads_dim)
        # out: (N, query_len, heads, head_dim) then flatten last two dimensions

        out = self.fc_out(out)
        return out


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


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_hid, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dropout = 0) -> None:
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
    def __init__(self, vocab_size, embed_dim, num_heads, d_hid, n_layers, dropout) -> None:
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
        self.softmax = nn.Softmax(dim = -1)
        self.positional_enc = PositionalEncoding(embed_dim)

    def forward(self, x, src_mask, trg_mask):
        # input shape: (batch_size, seq_len)

        # Embed the inputs and targets
        enc_out = self.enc_embedding(x)
        dec_out = self.dec_embedding(x)

        # Apply the positional encodings
        enc_out = self.positional_enc(enc_out)
        dec_out = self.positional_enc(dec_out)

        # Run the transformer layers
        for layer in self.enc_layers:
            enc_out = layer(enc_out, src_mask)

        for layer in self.dec_layers:
            dec_out = layer(dec_out, enc_out, trg_mask)

        # Apply final linear layer and softmax to make predictions
        y = self.fc_out(dec_out)
        y = self.softmax(y)

        return y


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    target = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    embed_dim = 200
    num_heads = 8
    d_hid = 200
    n_layers = 1
    vocab_size = 300

    def create_masks(x, batch_size, max_len, num_heads):
        trg_mask = torch.triu(torch.ones(batch_size, num_heads, max_len, max_len), diagonal = 1) # For target mask
        src_mask = torch.where(trg_mask == 1, 0, 1)
        return src_mask, src_mask


    def tokenizer(sentence):
        return list(sentence.lower())

    start_time = time.time()

    train_iter = WikiText2(split = 'train')
    #tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials = ['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    print(len(vocab))

    def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(vocab(tokenizer(item)), dtype = torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def batchify(data: Tensor, bsz: int) -> Tensor:
        """Divides the data into bsz separate sequences, removing extra elements
        that wouldn't cleanly fit.

        Args:
            data: Tensor, shape [N]
            bsz: int, batch size

        Returns:
            Tensor of shape [N // bsz, bsz]
        """
        seq_len = data.size(0) // bsz
        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data.to(device)

    batch_size = 500
    eval_batch_size = 500
    train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)

    bptt = 100
    def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
        """
        Args:
            source: Tensor, shape [full_seq_len, batch_size]
            i: int

        Returns:
            tuple (data, target), where data has shape [seq_len, batch_size] and
            target has shape [seq_len * batch_size]
        """
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].reshape(-1)

        return data, target

    warmup_steps = 5
    ntokens = len(vocab)  # size of vocabulary
    emsize = 200  # embedding dimension # d_model
    d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 1  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 8  # number of heads in nn.MultiheadAttention
    dropout = 0  # dropout probability
    model = Generator(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 0 #1e-5  # learning rate
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 0, betas = (0.9, 0.98), eps = 1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma = 0.95)
    lookup = vocab.get_stoi()

    def train(model: nn.Module) -> None:
        model.train()  # turn on train mode
        total_loss = 0.
        log_interval = 100
        start_time = time.time()

        num_batches = len(train_data) // bptt
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            data = data.reshape(data.shape[1], data.shape[0])
            current_batch_size = data.shape[0]

            src_mask, trg_mask = create_masks(data, current_batch_size, data.shape[1], num_heads)
            src_mask = src_mask.to(device)
            trg_mask = trg_mask.to(device)

            output = model(data, src_mask, trg_mask)
            loss = criterion(output.view(-1, ntokens), targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if batch % log_interval == 0 and batch > 0:
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                ppl = math.exp(cur_loss)
                print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.7f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                total_loss = 0
                start_time = time.time()

        if epoch % 1 == 0:
            input_string = 'there are many things about horses that have been discovered in recent '
            
            int_to_word = vocab.get_itos() # Array of vocab words
            sentence = ''
            word = ''
            i = 0
            unk_idx = lookup['<unk>']

            with torch.no_grad():
                while True == True:
                    i += 1

                    tokens = tokenizer(input_string)

                    input_tensor = torch.tensor([vocab(tokens)], dtype = torch.long).to(device)
                    src_mask = torch.ones((1, 1), dtype = torch.float).to(device)

                    output = model(input_tensor, src_mask, src_mask)

                    # Works the best
                    output = torch.sum(output, dim = 0)
                    output = output[-1]
                    top_k = torch.topk(output, 2, dim = 0).indices

                    if top_k[0].item() != unk_idx:
                        word_idx = top_k[0].item()
                    else:
                        word_idx = top_k[1].item()

                    word = int_to_word[word_idx]
                    sentence += word
                    #sentence += ' '
                    input_string += word
                    #input_string += ' '

                    if i > 100: #25
                        break

            print(input_string)
            print('Sentence: ', sentence)

    def evaluate(model: nn.Module, eval_data: Tensor) -> float:
        model.eval()  # turn on evaluation mode
        total_loss = 0.
        src_mask = generate_square_subsequent_mask(bptt).to(device)
        with torch.no_grad():
            for i in range(0, eval_data.size(0) - 1, bptt):
                data, targets = get_batch(train_data, i)
                data = data.reshape(data.shape[1], data.shape[0])
                current_batch_size = data.shape[0]

                src_mask, trg_mask = create_masks(data, current_batch_size, data.shape[1], num_heads)
                src_mask = src_mask.to(device)
                trg_mask = trg_mask.to(device)

                output = model(data, src_mask, trg_mask)
                output_flat = output.view(-1, ntokens)
                total_loss += batch_size * criterion(output_flat, targets).item()
        return total_loss / (len(eval_data) - 1)

    best_val_loss = float('inf')
    epochs = 5
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # Change the learning rate according to warmup step formula
        for g in optimizer.param_groups:
            g['lr'] = (1 / math.sqrt(emsize)) * min((1 / math.sqrt(epoch)), epoch * (1 / math.sqrt(warmup_steps ** 3)))

        train(model)
        val_loss = evaluate(model, val_data)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        #scheduler.step()

    test_loss = evaluate(best_model, test_data)
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | '
        f'test ppl {test_ppl:8.2f}')
    print('=' * 89)

    torch.save(model, 'text-generator.pickle')

    print('Finished in %s seconds' % str(round(time.time() - start_time, 2)))