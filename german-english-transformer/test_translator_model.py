from utils import translate_sentence_en, translate_sentence_en, bleu, save_checkpoint, load_checkpoint
import torch
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import torch.nn as nn
from models import Transformer

import torch.optim as optim
import spacy


print("Brabbbleeee")

spacy_ger = spacy.load("de")
spacy_eng = spacy.load("en")


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(
    tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".en", ".de"), fields=(english, german)
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = False

# Training hyperparameters
num_epochs = 50
learning_rate = 3e-4
batch_size = 32

# Model hyperparameters
src_vocab_size = len(english.vocab)
trg_vocab_size = len(german.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4
src_pad_idx = german.vocab.stoi["<pad>"]

load_model = True

model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

