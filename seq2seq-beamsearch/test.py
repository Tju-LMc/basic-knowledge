from torchtext.experimental.datasets.raw import Multi30k
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import Counter
from torchtext.vocab import Vocab
from itertools import tee
import torch
from torch.nn.utils.rnn import pad_sequence

EPOCHS = 10 # epoch
LR = 5  # learning rate
BATCH_SIZE = 64 # batch size for training

de_tokenizer = get_tokenizer('spacy', language='de')
en_tokenizer = get_tokenizer('spacy', language='en')

# train_dataset, valid_dataset, test_dataset = Multi30k(tokenizer=(de_tokenizer, en_tokenizer))
train_dataset, valid_dataset, test_dataset = Multi30k()
train_dataset1, train_dataset2  = tee(train_dataset)
valid_dataset1, valid_dataset2  = tee(valid_dataset)
test_dataset1, test_dataset2  = tee(test_dataset)

de_counter = Counter()
en_counter = Counter()
def build_vocab(dataset):
  # for idx, (src_sentence, tgt_sentence) in tqdm(enumerate(dataset)):
  for (src_sentence, tgt_sentence) in tqdm(dataset):
    de_counter.update(de_tokenizer(src_sentence))
    en_counter.update(en_tokenizer(tgt_sentence))

build_vocab(train_dataset1)
build_vocab(valid_dataset1)
build_vocab(test_dataset1)

de_vocab = Vocab(de_counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
en_vocab = Vocab(en_counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

PAD_IDX = de_vocab.stoi['<pad>']
BOS_IDX = de_vocab.stoi['<bos>']
EOS_IDX = de_vocab.stoi['<eos>']

def data_process(dataset):
  data = []
  for (raw_de, raw_en) in tqdm(dataset):
    de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de)], dtype=torch.long)
    en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)], dtype=torch.long)
    data.append((de_tensor_, en_tensor_))
  return data

train_data = data_process(train_dataset2)
val_data = data_process(valid_dataset2)
test_data = data_process(test_dataset2)

def generate_batch(data_batch):
  de_batch, en_batch = [], []
  for (de_item, en_item) in data_batch:
    de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
  de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
  en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
  return de_batch, en_batch

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)


for idx, batch in enumerate(train_dataloader):
  src, trg = batch
  print(batch)