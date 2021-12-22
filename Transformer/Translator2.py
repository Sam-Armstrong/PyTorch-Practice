# Some of this code is reused from the PyTorch website

import math
import copy
import time
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import spacy


class TransformerModel(nn.Module):

    def __init__(self, src_ntoken: int, trg_ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

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


if __name__ == '__main__':
    start_time = time.time()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bptt = 35

    spacy_ger = spacy.load('de_core_news_sm')
    spacy_eng = spacy.load('en_core_web_sm')

    def tokenize_ger(text):
        return [tok.text for tok in spacy_ger.tokenizer(text)]

    def tokenize_eng(text):
        return [tok.text for tok in spacy_eng.tokenizer(text)]

    train_iter = Multi30k(split = 'train', language_pair = ('de', 'en'))
    #tokenizer = get_tokenizer('basic_english')
    german_vocab = build_vocab_from_iterator(map(tokenize_ger, train_iter), specials=['<unk>'])
    german_vocab.set_default_index(german_vocab['<unk>'])
    train_iter = Multi30k(split = 'train', language_pair = ('de', 'en'))
    english_vocab = build_vocab_from_iterator(map(tokenize_eng, train_iter), specials=['<unk>'])
    english_vocab.set_default_index(english_vocab['<unk>'])

    def data_process_german(raw_text_iter: dataset.IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(german_vocab(tokenize_ger(item)), dtype = torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def data_process_english(raw_text_iter: dataset.IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [torch.tensor(english_vocab(tokenize_eng(item)), dtype = torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

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
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target

    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = Multi30k(language_pair = ('de', 'en'))
    # train_data = data_process(train_iter)
    # val_data = data_process(val_iter)
    # test_data = data_process(test_iter)

    batch_size = 50 #20
    eval_batch_size = 50 #10
    train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)

    src_ntokens = len(german_vocab)  # size of vocabulary
    trg_ntokens = len(english_vocab)
    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    model = TransformerModel(src_ntokens, trg_ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 3e-5 #5.0  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # SGD
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    def train(model: nn.Module) -> None:
        model.train()  # turn on train mode
        total_loss = 0.
        log_interval = 500
        start_time = time.time()
        src_mask = generate_square_subsequent_mask(bptt).to(device)

        num_batches = len(train_data) // bptt
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
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
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.6f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f}')
                total_loss = 0
                start_time = time.time()

    def evaluate(model: nn.Module, eval_data: Tensor) -> float:
        model.eval()  # turn on evaluation mode
        total_loss = 0.
        src_mask = generate_square_subsequent_mask(bptt).to(device)
        with torch.no_grad():
            for i in range(0, eval_data.size(0) - 1, bptt):
                data, targets = get_batch(eval_data, i) # data shape: (35, 10) (bptt, batch_size)
                batch_size = data.size(0)
                if batch_size != bptt:
                    src_mask = src_mask[:batch_size, :batch_size]
                output = model(data, src_mask)
                output_flat = output.view(-1, ntokens)
                total_loss += batch_size * criterion(output_flat, targets).item()
        return total_loss / (len(eval_data) - 1)

    best_val_loss = float('inf')
    epochs = 3
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model)
        val_loss = evaluate(model, val_data)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {val_loss:5.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        #scheduler.step()

    
    torch.save(model, 'model.pickle')

    print('Time to finish: ', round(time.time() - start_time, 2), ' seconds')

    cancelled = False
    lookup = vocab.get_stoi()

    while cancelled == False:
        input_string = input('Input some text: ')

        tokens = tokenizer(input_string)
        start_point = len(tokens) - bptt
        input_tensor = torch.zeros((len(tokens), 1), dtype = torch.long).to(device) #35
        src_mask = generate_square_subsequent_mask(len(tokens)).to(device) #bptt

        for i in range(len(tokens)):
            if i >= start_point:
                input_tensor[i][0] = lookup[tokens[i]]
                #input_tensor[i - start_point][0] = vocab(tokens[i])

        output = model(input_tensor, src_mask)
        output = torch.argmax(output, dim = -1)
        output = output.reshape(output.shape[0])

        #print(output)
        #print(output.shape)

        int_to_word = vocab.get_itos() # Array of vocab words

        sentence = ''
        for i in range(output.shape[0]):
            sentence += int_to_word[output[i].item()]
            sentence += ' '

        print(sentence)