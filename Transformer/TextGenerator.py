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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout) ##
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation = nn.functional.selu)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)
        self.softmax = nn.Softmax(dim = -1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

        self.encoder.weight.data.normal_(0, 1 / math.sqrt(emsize))
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-(math.sqrt(6) / math.sqrt(emsize + d_hid)), math.sqrt(6) / math.sqrt(emsize + d_hid))
        # (-(math.sqrt(6) / math.sqrt(200 + 200)), math.sqrt(6) / math.sqrt(200 + 200))

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, emsize]
        """
        #print(src)
        src = self.encoder(src) * math.sqrt(self.d_model) # *
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        #output = self.softmax(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    x = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal = 1)
    #x[:, -20:-1] = 0
    return x


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

batch_size = 300
bptt = 200
warmup_steps = 20 #200
ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension # d_model
d_hid = 250  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 5  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 5  # number of heads in nn.MultiheadAttention
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

model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

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
        # print(output.shape)
        # print(output.view(-1, ntokens).shape)
        # print(targets.shape)
        # output = torch.argmax(output, dim = -1)
        # loss = criterion(output.view(-1, ntokens), targets)
        # output = output.float()
        # targets = targets.float()
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
        input_string = 'there are many things about horses that have been discovered in recent'
        #input_string = 'in a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the andes mountains. even more surprising to the researchers was the fact that the unicorns spoke perfect english. '
        #input_string = 'in a shocking finding, scientists discovered a herd of unicorns living in a remote, previously explored valley'
        
        int_to_word = vocab.get_itos() # Array of vocab words
        sentence = ''
        word = ''
        i = 0
        #unk_idx = lookup['<unk>']

        #start_point = len(tokens) - bptt
        with torch.no_grad():
            while True == True: #word != '.':
                i += 1

                tokens = tokenizer(input_string)

                input_tensor = torch.tensor(vocab(tokens), dtype = torch.long).to(device) # [vocab(tokens)]
                # input tensor shape: (71)
                input_tensor = nn.functional.pad(input = input_tensor, pad = (bptt - input_tensor.shape[0], 0), mode = 'constant', value = 0)
                # input tensor shape: (bptt)
                #src_mask = generate_square_subsequent_mask(bptt).to(device)
                #src_mask = torch.zeros((1, 1), dtype = torch.float).to(device) #ones # Works best
                src_mask = torch.zeros((bptt, bptt), dtype = torch.float).to(device)
                #print(src_mask.shape)

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
                ##

                # if top_k[0].item() != unk_idx:
                #     word_idx = top_k[0].item()
                # else:
                #     word_idx = top_k[1].item()

                word_idx = top_k[0].item()

                # Performs worse
                # output = torch.sum(output, dim = 0)
                # output = torch.sum(output, dim = 0)
                # word_idx = torch.argmax(output, dim = 0).item()

                word = int_to_word[word_idx]
                sentence += word
                #sentence += ' '
                input_string += word
                #input_string += ' '

                # if i == 1:
                #     print(torch.max(output).item())

                if i > 125: #25
                    break

        print(input_string)
        print('Sentence: ', sentence)

        #output = torch.argmax(output, dim = -1)

        # sentence = ''
        # for i in range(output.shape[0]):
        #     word = int_to_word[output[i, -1].item()]
        #     sentence += word
        #     #sentence += ' '

        # print('')

        # output = torch.flatten(output) #

        # sentence = ''
        # for i in range(output.shape[0]):
        #     word = int_to_word[output[i].item()]
        #     #word = int_to_word[output[i, -1].item()]
        #     if '.' in word:
        #         sentence += '.'
        #         break
        #     sentence += word
        #     #sentence += ' '

        # print(sentence)
        # print(len(sentence))

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
        g['lr'] = 1 * (1 / math.sqrt(emsize)) * min((1 / math.sqrt(epoch)), epoch * (1 / math.sqrt(warmup_steps ** 3)))
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