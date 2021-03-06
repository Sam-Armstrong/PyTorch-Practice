import math
from typing import Tuple
import copy
import time

import torch
from torch import Tensor
import torch.optim as optim
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torchtext.datasets import WikiText2, WikiText103
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal = 1)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads, dropout = 0) -> None:
        super(SelfAttention, self).__init__()

        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        assert (self.head_dim * heads == embed_dim), 'Embed size needs to be divisible by heads'

        # Switch to head_dim -> head_dim
        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.Linear(self.embed_dim, embed_dim)
        #print(embed_dim)

        self.values_embed = nn.Embedding(self.embed_dim, self.heads)
        self.keys_embed = nn.Embedding(self.embed_dim, self.heads)
        self.queries_embed = nn.Embedding(self.embed_dim, self.heads)
        

    def forward(self, values, keys, queries, src_mask):
        N = queries.shape[1]
        value_len, key_len, query_len = values.shape[0], keys.shape[0], queries.shape[0]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])
        # queries: (N, query_len, heads, head_dim)
        # keys: (N, key_len, heads, head_dim)
        # energy: (N, heads, query_len, key_len)

        if src_mask is not None:
            energy = energy.masked_fill(src_mask == 0, float('-1e20'))

        attention = torch.softmax(energy / (self.embed_dim ** (1/2)), dim = 3)

        out = torch.einsum('nhql,nlhd->nqhd', [attention, values]).reshape(query_len, N, self.heads * self.head_dim) # Performs the matmul then concatenates
        # attention: (N, heads, query_len, key_len)
        # values: (N, value_len, heads, heads_dim)
        # out: (N, query_len, heads, head_dim) then flatten last two dimensions

        out = self.fc_out(out)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, d_hid, dropout = 0) -> None: # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super(EncoderLayer, self).__init__()

        self.attention = SelfAttention(embed_dim, heads)
        self.fc1 = nn.Linear(embed_dim, d_hid)
        self.fc2 = nn.Linear(d_hid, embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.activation =  nn.SELU()

    def forward(self, values, keys, queries, src_mask):
        # values = values.reshape(values.shape[1], values.shape[0], values.shape[2])
        # keys = keys.reshape(keys.shape[1], keys.shape[0], keys.shape[2])
        # queries = queries.reshape(queries.shape[1], queries.shape[0], queries.shape[2])

        res = values.clone()
        x = self.attention(values, keys, queries, src_mask)
        x += res
        x = self.ln1(x)

        res = x.clone()
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x += res
        x = self.ln2(x)
        
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000, batch_size = 10):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        self.pe = torch.zeros(max_len, 1, embed_dim) # 1
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)
        #self.pe = self.pe.repeat(1, batch_size, 1)
        #self.register_buffer('pe', self.pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        print(x) ###Issue here
        print(x.shape)
        print(self.pe[:x.shape[0]].shape)
        print(x)
        print(self.pe[:x.shape[0]])
        #x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TextGen(nn.Module):
    def __init__(self, embed_dim, heads, d_hid, n_layers, vocab_size, seq_len, dropout = 0) -> None:
        super(TextGen, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) # (seq_len, embed_dim)
        self.pe = nn.Embedding(seq_len, embed_dim)
        self.layers = [EncoderLayer(embed_dim, heads, d_hid, dropout = dropout).to(device) for _ in range(n_layers)]
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.softmax = nn.Softmax(dim = -1)
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, src_mask):
        #print(x)
        #print(x.shape)

        # x shape: (seq_len, batch_size)
        # Duplicates the seqence into multiple rows to allow subseqeunt masking
        #x = x.reshape(x.shape[0], x.shape[1], 1)
        #x = x.repeat(1, 1, self.embed_dim)
        # Required shape for x: (seq_len, batch_size, embed_dim)

        #x = x.reshape(x.shape[1], x.shape[0]) # (batch_size, seq_len)
        batch_size = x.shape[1]
        seq_len = x.shape[0]

        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(self.device)
        positions = positions.reshape(positions.shape[1], positions.shape[0])

        x = self.embedding(x) / math.sqrt(self.embed_dim)
        x += self.pe(positions) # Add the positional encoding
        # x shape: (200, 10, 400)

        #print(x)

        for layer in self.layers:
            x = layer(x, x, x, src_mask)

        x = self.fc_out(x)
        x = self.softmax(x)

        return x

if __name__ == '__main__':
    def tokenizer(sentence):
        return list(sentence.lower())

    start_time = time.time()

    train_iter = WikiText2(split = 'train')  #WikiText103(split = 'train') #WikiText2(split = 'train') 
    #tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter))#, specials = ['<unk>'])
    vocab.set_default_index(vocab['a'])
    print(len(vocab))

    def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(vocab(tokenizer(item)), dtype = torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2() #WikiText103() #WikiText2()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

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

    batch_size = 100
    bptt = 200
    warmup_steps = 200 #200
    ntokens = len(vocab)  # size of vocabulary
    emsize = 400  # embedding dimension # d_model
    d_hid = 400  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 1  # number of heads in nn.MultiheadAttention
    dropout = 0.1  # dropout probability


    train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
    val_data = batchify(val_data, batch_size)
    test_data = batchify(test_data, batch_size)

    # train_mean = torch.mean(train_data, dim = 1) ##
    # train_std = torch.std(train_data, dim = 1)

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

    model = TextGen(emsize, nhead, d_hid, nlayers, ntokens, bptt, dropout = 0).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 1e-5  # learning rate
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 0, betas = (0.9, 0.98), eps = 1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma = 0.95)
    lookup = vocab.get_stoi()
    #sm = nn.Softmax(dim = -1)

    def train(model: nn.Module) -> None:
        model.train()  # turn on train mode
        total_loss = 0.
        log_interval = 100
        start_time = time.time()
        src_mask = generate_square_subsequent_mask(bptt).to(device)

        num_batches = len(train_data) // bptt
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            # for g in optimizer.param_groups:
            #     step_num = (epoch * train_data.size(0)) + batch # num_batches
            #     print(0.05 * (1 / math.sqrt(emsize)) * min((1 / math.sqrt(step_num)), step_num * (1 / math.sqrt(warmup_steps ** 3))))
            #     g['lr'] = 0.05 * (1 / math.sqrt(emsize)) * min((1 / math.sqrt(step_num)), step_num * (1 / math.sqrt(warmup_steps ** 3)))

            data, targets = get_batch(train_data, i)
            batch_size = data.size(0)
            if batch_size != bptt:  # only on last batch
                src_mask = src_mask[:batch_size, :batch_size]

            output = model(data, src_mask)
            loss = criterion(output.view(-1, ntokens), targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if batch % log_interval == 0 and batch > 0:
                #lr = scheduler.get_last_lr()[0]
                lr = get_lr(optimizer)
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                ppl = math.exp(cur_loss)
                print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.7f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                total_loss = 0
                start_time = time.time()

        if epoch % 1 == 0:
            model.eval()
            input_string = 'there are many things about horses that have been discovered in recent '
            #input_string = 'in a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the andes mountains. even more surprising to the researchers was the fact that the unicorns spoke perfect english. '
            #input_string = 'in a shocking finding, scientists discovered a herd of unicorns living in a remote, previously explored valley'
            
            int_to_word = vocab.get_itos() # Array of vocab words
            sentence = ''
            word = ''
            i = 0

            with torch.no_grad():
                while True == True:
                    i += 1

                    tokens = tokenizer(input_string)

                    input_tensor = torch.tensor(vocab(tokens), dtype = torch.long).to(device)
                    # input tensor shape: (71)
                    input_tensor = nn.functional.pad(input = input_tensor, pad = (bptt - input_tensor.shape[0], 0), mode = 'constant', value = 0)
                    src_mask = torch.zeros((bptt, bptt), dtype = torch.float).to(device)

                    # src: Tensor, shape [seq_len, batch_size]
                    # src_mask: Tensor, shape [seq_len, seq_len]

                    input_tensor = input_tensor.reshape(bptt, 1)

                    output = model(input_tensor, None)
                    #print(output.shape)
                    
                    # output shape: (bptt, 1, n_tokens)
                    # output shape should be: [seq_len, batch_size, ntoken]

                    # Works the best
                    #output = torch.sum(output, dim = 0)
                    # output = output[-1, -1]
                    # top_k = torch.topk(output, 2, dim = 0).indices

                    # Experiment
                    #output = torch.sum(output, dim = 0)
                    output = output[-1]
                    output = torch.sum(output, dim = 0)
                    top_k = torch.topk(output, 2, dim = 0).indices
                    word_idx = top_k[0].item()

                    word = int_to_word[word_idx]
                    sentence += word
                    #sentence += ' '
                    input_string += word
                    #input_string += ' '

                    if i > 125:
                        break

            print(input_string)
            print('Sentence: ', sentence)

    def evaluate(model: nn.Module, eval_data: Tensor) -> float:
        model.eval()  # turn on evaluation mode
        total_loss = 0.
        src_mask = generate_square_subsequent_mask(bptt).to(device)
        with torch.no_grad():
            for i in range(0, eval_data.size(0) - 1, bptt):
                data, targets = get_batch(eval_data, i)
                batch_size = data.size(0)
                if batch_size != bptt:
                    src_mask = src_mask[:batch_size, :batch_size]
                output = model(data, src_mask)
                output_flat = output.view(-1, ntokens)
                total_loss += batch_size * criterion(output_flat, targets).item()
        return total_loss / (len(eval_data) - 1)

    best_val_loss = float('inf')
    epochs = 5000
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # Change the learning rate according to warmup step formula
        for g in optimizer.param_groups:
            g['lr'] = 0.1 * (1 / math.sqrt(emsize)) * min((1 / math.sqrt(epoch)), epoch * (1 / math.sqrt(warmup_steps ** 3)))
            #g['weight_decay'] = 0.005 * (1 / math.sqrt(emsize)) * min((1 / math.sqrt(epoch)), epoch * (1 / math.sqrt(warmup_steps ** 3)))

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