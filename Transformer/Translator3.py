import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import dataset
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import math
import copy
import time
from typing import Tuple
import spacy
from torch.utils.data import DataLoader

class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        heads,
        n_encoder_layers,
        n_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)
        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            heads,
            n_encoder_layers,
            n_decoder_layers,
            forward_expansion,
            dropout,
        )

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        # src shape: (src_len, N)
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        # mask shape: (N, src_len)
        return src_mask

    def forward(self, src, trg):
        src_seq_len, N_src = src.shape
        trg_seq_len, N_trg = trg.shape

        src_positions = (torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, N_src).to(self.device))
        trg_positions = (torch.arange(0, trg_seq_len).unsqueeze(1).expand(trg_seq_len, N_trg).to(self.device))
        embed_src = self.dropout((self.src_word_embedding(src) + self.src_position_embedding(src_positions)))
        embed_trg = self.dropout((self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)))

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(self.device)

        out = self.transformer(embed_src, embed_trg, src_key_padding_mask = src_padding_mask, tgt_mask = trg_mask)
        out = self.fc_out(out)
        return out


if __name__ == '__main__':
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 100

    # Preprocessing
    spacy_ger = spacy.load('de_core_news_sm')
    spacy_eng = spacy.load('en_core_web_sm')


    def tokenize_ger(text):
        if type(text) == tuple:
            tokens = [tok.text for tok in spacy_ger.tokenizer(text[0])]
        else:
            tokens = [tok.text for tok in spacy_ger.tokenizer(text)]
        return tokens

    def tokenize_eng(text):
        if type(text) == tuple:
            tokens = [tok.text for tok in spacy_eng.tokenizer(text[1])]
        else:
            tokens = [tok.text for tok in spacy_eng.tokenizer(text)]
        return tokens

    # def tokenize_ger(text):
    #     if type(text) == tuple:
    #         tokens = list(text[0])
    #     else:
    #         tokens = list(text)
    #     return tokens

    # def tokenize_eng(text):
    #     if type(text) == tuple:
    #         tokens = list(text[1])
    #     else:
    #         tokens = list(text)
    #     return tokens

    train_iter = Multi30k(split = 'train', language_pair = ('de', 'en'))
    german_vocab = build_vocab_from_iterator(map(tokenize_ger, train_iter), specials = ['<unk>', '<pad>'])
    german_vocab.set_default_index(german_vocab['<unk>'])
    train_iter = Multi30k(split = 'train', language_pair = ('de', 'en'))
    english_vocab = build_vocab_from_iterator(map(tokenize_eng, train_iter), specials = ['<unk>', '<pad>'])
    english_vocab.set_default_index(english_vocab['<unk>'])

    print(len(german_vocab))
    print(len(english_vocab))
    #print(german_vocab.get_itos())

    load_model = False
    save_model = True
    epochs = 15
    lr = 3e-1

    src_vocab_size = len(german_vocab)
    trg_vocab_size = len(english_vocab)
    embedding_size = 128 #512
    heads = 1
    n_encoder_layers = 4
    n_decoder_layers = 4
    dropout = 0.1
    max_len = 100
    forward_expansion = 128
    src_pad_idx = english_vocab.get_stoi()['<pad>']

    # Not using one-hot encoding
    def data_process(raw_text):
        ger_data = torch.zeros((len(raw_text), max_len), dtype = torch.long).to(device)
        eng_data = torch.zeros((len(raw_text), max_len), dtype = torch.long).to(device)
        g_stoi = german_vocab.get_stoi()
        e_stoi = english_vocab.get_stoi()
        
        for i, (ger, eng) in enumerate(raw_text):
            ger_start = max_len - len(tokenize_ger(ger))
            eng_start = max_len - len(tokenize_eng(eng))

            for j, tok in enumerate(tokenize_ger(ger)):
                try:
                    ger_data[i, j + ger_start] = g_stoi[tok]
                except:
                    ger_data[i, j + ger_start] = g_stoi['<unk>']
            for j, tok in enumerate(tokenize_eng(eng)):
                try:
                    eng_data[i, j + eng_start] = e_stoi[tok]
                except:
                    eng_data[i, j + eng_start] = e_stoi['<unk>']

        return ger_data, eng_data

    # Using one-hot encoding
    # def data_process(raw_text):
    #     ger_data = torch.zeros((len(raw_text), max_len, len(german_vocab)), dtype = torch.long).to(device)
    #     eng_data = torch.zeros((len(raw_text), max_len, len(english_vocab)), dtype = torch.long).to(device)
    #     g_stoi = german_vocab.get_stoi()
    #     e_stoi = english_vocab.get_stoi()
        
    #     for i, (ger, eng) in enumerate(raw_text):
    #         for j, tok in enumerate(tokenize_ger(ger)):
    #             try:
    #                 ger_data[i, j, g_stoi[tok]] = 1
    #             except:
    #                 ger_data[i, j, g_stoi['<unk>']] = 1
    #         for j, tok in enumerate(tokenize_eng(eng)):
    #             try:
    #                 eng_data[i, j, e_stoi[tok]] = 1
    #             except:
    #                 eng_data[i, j, e_stoi['<unk>']] = 1

    #     return ger_data, eng_data

    def data_process_english(raw_text_iter: dataset.IterableDataset) -> Tensor:
        data = [torch.tensor(english_vocab(tokenize_eng(item[1])), dtype = torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    # def data_process_german(raw_text_iter: dataset.IterableDataset) -> Tensor:
    #     data = [torch.tensor(german_vocab(tokenize_ger(item[0])), dtype = torch.long) for item in raw_text_iter]
    #     return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_data, valid_data, test_data = Multi30k(split = ('train', 'valid', 'test'), language_pair = ('de', 'en'))

    ger_data, eng_data = data_process(train_data)
    eng_train_iter = DataLoader(eng_data, batch_size = batch_size)
    ger_train_iter = DataLoader(ger_data, batch_size = batch_size)

    ger_data, eng_data = data_process(valid_data)
    eng_valid_iter = DataLoader(eng_data, batch_size = batch_size)
    ger_valid_iter = DataLoader(ger_data, batch_size = batch_size)

    ger_data, eng_data = data_process(test_data)
    eng_test_iter = DataLoader(eng_data, batch_size = batch_size)
    ger_test_iter = DataLoader(ger_data, batch_size = batch_size)


    # Training
    model = Transformer(
        embedding_size,
        src_vocab_size, 
        trg_vocab_size, 
        src_pad_idx,
        heads,
        n_encoder_layers,
        n_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr = lr)
    pad_idx = english_vocab.get_stoi()['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

    # Load model?

    sentence = 'ich mag deinen roten hut'
    target_sentence = 'i like your red hat'

    for epoch in range(epochs):
        print('Epoch %s' % str(epoch + 1))

        model.train()

        for batch_idx, (inp_data, target) in enumerate(zip(ger_train_iter, eng_train_iter)):

            output = model(inp_data, target[:-1])
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            model.zero_grad()

            loss = criterion(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)

            optimizer.step()

        
        model.eval()

        tokens = tokenize_ger(sentence)
        inp = torch.empty((len(tokens), 1), dtype = torch.long).to(device)

        for j, tok in enumerate(tokens):
            try:
                inp[j] = german_vocab.get_stoi()[tok]
            except:
                inp[j] = german_vocab.get_stoi()['<unk>']
        
        tokens = tokenize_eng(target_sentence)
        trg = torch.zeros((len(tokens), 1), dtype = torch.long).to(device)
        # for j, tok in enumerate(tokens):
        #     try:
        #         trg[j] = english_vocab.get_stoi()[tok]
        #     except:
        #         trg[j] = english_vocab.get_stoi()['<unk>']

        output = model(inp, trg)
        output = output.reshape(-1, output.shape[2]) ##
        #print(torch.argmax(output[0, 0]))

        translated_sentence = torch.argmax(output, dim = -1)
        translated_sentence = translated_sentence.reshape(translated_sentence.shape[0])

        output_sentence = ''
        for word in range(translated_sentence.shape[0]):
            output_sentence += english_vocab.get_itos()[translated_sentence[word]]
            output_sentence += ' '

        print('Sentence: ', output_sentence)

    torch.save(model, 'transformer_model.pickle')

    print('Finished in ', round(time.time() - start_time, 2), ' seconds')
