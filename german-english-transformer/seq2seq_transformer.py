"""
Seq2Seq using Transformers on the Multi30k
dataset. In this video I utilize Pytorch
inbuilt Transformer modules, and have a
separate implementation for Transformers
from scratch. Training this model for a
while (not too long) gives a BLEU score
of ~35, and I think training for longer
would give even better results.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence_en, translate_sentence_en, bleu, save_checkpoint, load_checkpoint
from models import Transformer
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from nltk.tokenize.treebank import TreebankWordDetokenizer

"""
To install spacy languages do:
python -m spacy download en
python -m spacy download de
"""
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

# these are just to enable testing of loaded models without training etc
training_wanted = True
testing_wanted = False
load_model = False
save_model = True

# Training hyperparameters
num_epochs = 5
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

# Tensorboard to get nice loss plot
writer = SummaryWriter("runs/loss_plot")
step = 0

#this also does padding - so i get shape will be the max batch sentence length with padding for shorter sentences
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

# import the custom pytorch nn.Transformer from the models file
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

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = german.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("checkpoints/my_checkpoint.pth.tar"), model, optimizer)

sentence = "the man walked with a horse under a bridge."
# if we just want to test a loaded model have this set to false
if training_wanted:
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch} / {num_epochs}]")

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename="checkpoints/my_checkpoint_v4.pth.tar")

        model.eval()
        translated_sentence = translate_sentence_en(
            model, sentence, english, german, device, max_length=50
        )

        print(f"Translated example sentence: \n {translated_sentence}")
        model.train()
        losses = []
        running_batch_loss = 0

        for batch_idx, batch in enumerate(train_iterator):

            print("length of iterator is: ", len(train_iterator))

            # Get input and targets and get to cuda
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)

            # print("input data shape before running through nn is: ", inp_data.shape)
            # print("target data shape before running through nn is: ", target.shape)

            # Forward prop - for the target we essentially remove the last item and this
            # somewhat shifts the decoders input to the right, so it is predicting next word
            # essentially the target will be one time step ahead of input to decoder
            output = model(inp_data, target[:-1, :])

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it

            # print("output shape before reshaping is: ", output.shape)
            output = output.reshape(-1, output.shape[2])
            # print("output shape is: ", output.shape)

            # print("target shape before reshape is: ", target.shape)
            target = target[1:].reshape(-1)
            # print("target shape is: ", target.shape)


            optimizer.zero_grad()

            loss = criterion(output, target)
            losses.append(loss.item())

            running_batch_loss += loss.item()
            # print("running batch loss is : ",  running_batch_loss)

            # Back prop
            loss.backward()
            # Clip to avoid exploding gradient issues, makes sure grads are
            # within a healthy range
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            # Gradient descent step
            optimizer.step()

            # plot to tensorboard
            writer.add_scalar("Training loss", loss, global_step=step)
            step += 1

        mean_loss = sum(losses) / len(losses)
        print("average loss for current epoch was: ", mean_loss)

        mean_batch_loss = running_batch_loss/len(train_iterator)
        print("and the average batch loss for epoch is: ", mean_batch_loss)
        break
        scheduler.step(mean_loss)


#Testing
# running on entire test data takes a while
if testing_wanted:
    score = bleu(test_data[1:50], model, english, german, device)
    print(f"Bleu score {score * 100:.2f}")

# custom sentence test on loaded/latest model

translated_sentence = translate_sentence_en(
            model, "cheese is nice", english, german, device, max_length=50
        )

