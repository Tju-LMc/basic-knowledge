"""RNN Beam Search example

I refered the following code for Beam Search.
https://github.com/budzianowski/PyTorch-Beam-Search-Decoding
"""
import argparse
import math
import os
import time

import spacy
import torch
import torch.nn as nn
import torch.optim as optim
# from torchtext.experimental.datasets import Multi30k
from torchtext.experimental.datasets.raw import Multi30k
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import Counter
from torchtext.vocab import Vocab
from itertools import tee

from beam import beam_search_decoding, batch_beam_search_decoding
from models import EncoderRNN, DecoderRNN, Attention, AttnDecoderRNN, Seq2Seq

BATCH_SIZE = 64 # batch size for training

# # utils {{{
# def tokenize_de(text):
#     """
#     Tokenizes German text from a string into a list of strings (tokens) and reverses it
#     """
#     return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

# def tokenize_en(text):
#     """
#     Tokenizes English text from a string into a list of strings (tokens)
#     """
#     return [tok.text for tok in spacy_en.tokenizer(text)]

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_n_best(decoded_seq, itos):
    for rank, seq in enumerate(decoded_seq):
        print(f'Out: Rank-{rank+1}: {" ".join([itos[idx] for idx in seq])}')
# }}}

# train {{{
def train(model, itr, optimizer, criterion):
    print('Start training')
    model.train()
    epoch_loss = 0
    for batch in tqdm(itr):
        # src = batch.src # (T, bs)
        # trg = batch.trg # (T, bs)
        src, trg = batch

        optimizer.zero_grad()

        output = model(src, trg)

        output_size = output.shape[-1]

        output = output[1:].view(-1, output_size)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(itr)
# }}}

# evaluate {{{
def evaluate(model, itr, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in itr:
            src = batch.src
            trg = batch.trg

            output = model(src, trg, teacher_forcing_ratio=0)

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(itr)
# }}}

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def main():
    # ArgumentParser {{{
    parser = argparse.ArgumentParser()
    # hyper parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--enc_embd_size', type=int, default=256)
    parser.add_argument('--dec_embd_size', type=int, default=256)
    parser.add_argument('--enc_h_size', type=int, default=512)
    parser.add_argument('--dec_h_size', type=int, default=512)
    # other parameters
    parser.add_argument('--beam_width', type=int, default=3)
    parser.add_argument('--n_best', type=int, default=5)
    parser.add_argument('--max_dec_steps', type=int, default=1000)
    parser.add_argument('--export_dir', type=str, default='./ckpts/')
    parser.add_argument('--model_name', type=str, default='s2s')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--attention', action='store_true')
    opts = parser.parse_args()
    # }}}


    # SOS_token = '<SOS>'
    # EOS_token = '<EOS>'
    # SRC = Field(tokenize=tokenize_de,
    #             init_token=SOS_token,
    #             eos_token=EOS_token,
    #             lower=True)
    # TRG = Field(tokenize=tokenize_en,
    #             init_token=SOS_token,
    #             eos_token=EOS_token,
    #             lower=True)
    # train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
    # print(f'Number of training examples: {len(train_data.examples)}')
    # print(f'Number of validation examples: {len(valid_data.examples)}')
    # print(f'Number of testing examples: {len(test_data.examples)}')

    # SRC.build_vocab(train_data, min_freq=2)
    # TRG.build_vocab(train_data, min_freq=2)
    # print(f'Unique tokens in source (de) vocabulary: {len(SRC.vocab)}')
    # print(f'Unique tokens in target (en) vocabulary: {len(TRG.vocab)}')
    # train_itr, valid_itr, test_itr =\
    #         BucketIterator.splits(
    #             (train_data, valid_data, test_data),
    #             batch_size=opts.batch_size,
    #             device=DEVICE)


    de_tokenizer = get_tokenizer('spacy', language='de')
    en_tokenizer = get_tokenizer('spacy', language='en')
    train_dataset, valid_dataset, test_dataset = Multi30k()
    train_dataset1, train_dataset2  = tee(train_dataset)
    valid_dataset1, valid_dataset2  = tee(valid_dataset)
    test_dataset1, test_dataset2  = tee(test_dataset)
    
    de_counter = Counter()
    en_counter = Counter()
    def build_vocab(dataset):
        for (src_sentence, tgt_sentence) in tqdm(dataset):
            de_counter.update(de_tokenizer(src_sentence))
            en_counter.update(en_tokenizer(tgt_sentence))

    build_vocab(train_dataset1)
    build_vocab(valid_dataset1)
    build_vocab(test_dataset1)
    de_vocab = Vocab(de_counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    en_vocab = Vocab(en_counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

    dec_v_size = len(de_vocab)
    enc_v_size = len(en_vocab)

    TRG_PAD_IDX = en_vocab.stoi['<pad>']
    TRG_SOS_IDX = en_vocab.stoi['<bos>']
    TRG_EOS_IDX = en_vocab.stoi['<eos>']

    def data_process(dataset):
        data = []
        for (raw_de, raw_en) in tqdm(dataset):
            de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de)], dtype=torch.long)
            en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)], dtype=torch.long)
            data.append((de_tensor_, en_tensor_))
        return data

    train_data = data_process(train_dataset2)
    valid_data = data_process(valid_dataset2)
    test_data = data_process(test_dataset2)

    def generate_batch(data_batch):
        de_batch, en_batch = [], []
        for (de_item, en_item) in data_batch:
            de_batch.append(torch.cat([torch.tensor([TRG_SOS_IDX]), de_item, torch.tensor([TRG_EOS_IDX])], dim=0))
            en_batch.append(torch.cat([torch.tensor([TRG_SOS_IDX]), en_item, torch.tensor([TRG_EOS_IDX])], dim=0))
        de_batch = pad_sequence(de_batch, padding_value=TRG_PAD_IDX)
        en_batch = pad_sequence(en_batch, padding_value=TRG_PAD_IDX)
        return de_batch, en_batch

    train_itr = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
    valid_itr = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
    test_itr = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)


    encoder = EncoderRNN(opts.enc_embd_size, opts.enc_h_size, opts.dec_h_size, enc_v_size, DEVICE)

    if opts.attention:
        attn = Attention(opts.enc_h_size, opts.dec_h_size)
        decoder = AttnDecoderRNN(opts.dec_embd_size, opts.enc_h_size, opts.dec_h_size, dec_v_size, attn, DEVICE)
    else:
        decoder = DecoderRNN(opts.dec_embd_size, opts.dec_h_size, dec_v_size,  DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    # TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    # TRG_PAD_IDX = tgt_vocab.stoi['<pad>']

    if opts.model_path != '':
        model.load_state_dict(torch.load(opts.model_path))

    if not opts.skip_train:
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
        best_valid_loss = float('inf')
        for epoch in range(opts.n_epochs):
            start_time = time.time()

            train_loss = train(model, train_itr, optimizer, criterion)
            valid_loss = evaluate(model, valid_itr, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                attn_type = 'attn' if opts.attention else 'vanilla'
                model_path = os.path.join(opts.export_dir, f'{opts.model_name}-{attn_type}.pt')
                print(f'Update model! Saved {model_path}')
                torch.save(model.state_dict(), model_path)
            else:
                print('Model was not updated. Stop training')
                break

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    # TRG_SOS_IDX = TRG.vocab.stoi[TRG.init_token]
    # TRG_EOS_IDX = TRG.vocab.stoi[TRG.eos_token]
    model.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(test_itr):
            src = batch.src # (T, bs)
            trg = batch.trg # (T, bs)
            print(f'In: {" ".join(SRC.vocab.itos[idx] for idx in src[:, 0])}')

            enc_outs, h = model.encoder(src) # (T, bs, H), (bs, H)
            # decoded_seqs: (bs, T)
            start_time = time.time()
            decoded_seqs = beam_search_decoding(decoder=model.decoder,
                                                enc_outs=enc_outs,
                                                enc_last_h=h,
                                                beam_width=opts.beam_width,
                                                n_best=opts.n_best,
                                                sos_token=TRG_SOS_IDX,
                                                eos_token=TRG_EOS_IDX,
                                                max_dec_steps=opts.max_dec_steps,
                                                device=DEVICE)
            end_time = time.time()
            print(f'for loop beam search time: {end_time-start_time:.3f}')
            print_n_best(decoded_seqs[0], TRG.vocab.itos)

            start_time = time.time()
            decoded_seqs = batch_beam_search_decoding(decoder=model.decoder,
                                                      enc_outs=enc_outs,
                                                      enc_last_h=h,
                                                      beam_width=opts.beam_width,
                                                      n_best=opts.n_best,
                                                      sos_token=TRG_SOS_IDX,
                                                      eos_token=TRG_EOS_IDX,
                                                      max_dec_steps=opts.max_dec_steps,
                                                      device=DEVICE)
            end_time = time.time()
            print(f'Batch beam search time: {end_time-start_time:.3f}')
            print_n_best(decoded_seqs[0], TRG.vocab.itos)


if __name__ == '__main__':
    main()
