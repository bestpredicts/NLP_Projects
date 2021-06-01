import functools
print = functools.partial(print, flush=True)
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
import datetime
import time, math
from Encoder import Encoder
from Decoder import Decoder
from Hyperparameters import args
import argparse
import pandas as pd
from datetime import date
# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

import argparse
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt


import torch
from torch import nn
import gensim
# import fasttext
from gensim.models import FastText
import random
import operator
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle
import joblib
from Hyperparameters import args
import nltk

from textdataMimic import TextDataMimic




textData = TextDataMimic("mimic", "../clinicalBERT/data/", "discharge", trainLM=False, test_phase=False, big_emb = False, new_emb = True)
        # self.start_token = self.textData.word2index['START_TOKEN']
        # self.end_token = self.textData.word2index['END_TOKEN']
args['vocabularySize'] = textData.getVocabularySize()
args['chargenum'] = 2
args['embeddingSize'] = textData.index2vector.shape[1]


class Vocabulary:
    """A vocabulary, assigns IDs to tokens"""

    def __init__(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w


class LSTM_Model(nn.Module):
    """
        LSTM
    """

    def __init__(self, w2i, i2w, LM, i2v=None, dimension=200):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(LSTM_Model, self).__init__()

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = 512

        # TODO try using the language model embedding after training!
        #         self.embedding = LM.embedding
        # self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(i2v))

        self.embedding = nn.Embedding(textData.getVocabularySize(), 200)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=200,
                            hidden_size=self.dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2 * dimension, 1)

    def forward(self, text, text_len):
        #         self.encoderInputs = encoderInputs.to(args['device'])
        #         self.encoder_lengths = encoder_lengths

        text_emb = self.embedding(text)
        text_len = text_len.cpu()

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out


# Save and Load Functions

def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model metrics saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model metrics loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


# define metric
def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(preds)

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


# Training Function
destination_folder = "./results/LSTM_Vanilla/"
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
device = args["device"]
args["batchSize"] = 128
eval_every = 50


def train(model,
          optimizer,
          criterion=nn.BCELoss(),
          num_epochs=5,
          file_path=destination_folder,
          best_valid_loss=float("Inf")):
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    batches = textData.getBatches()
    total_steps = num_epochs*len(batches)

    # training loop
    model.train()
    for epoch in range(num_epochs):
        print("starting epoch: ", epoch)

        for index, batch in enumerate(batches):
            x = {}
            x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
            x['enc_len'] = torch.LongTensor(batch.encoder_lens).to(args['device'])
            x['labels'] = autograd.Variable(torch.FloatTensor(batch.label)).to(args['device'])

            output = model(x['enc_input'], x['enc_len'])

            #             print(output)

            #             print(output.shape)
            #             print(x['labels'].shape)

            loss = criterion(output, x['labels'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            print(f"global step is: {global_step}")
            # evaluation step
            if global_step % eval_every == 0:
                print("About to run a validation run")

                model.eval()
                with torch.no_grad():
                    batches = textData.getBatches("dev")

                    #                   # validation loop
                    for index, batch in enumerate(batches):
                        x = {}
                        x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
                        x['enc_len'] = torch.LongTensor(batch.encoder_lens).to(args['device'])
                        x['labels'] = autograd.Variable(torch.FloatTensor(batch.label)).to(args['device'])

                        output = model(x['enc_input'], x['enc_len'])
                        loss = criterion(output, x['labels'])
                        valid_running_loss += loss.item()

                #                 # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(batches)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                #                 # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress

                print(f"Epoch number: {epoch}. Epoch loss: {average_train_loss}\n Val loss: {average_valid_loss}\n Step {global_step}/{total_steps}")
                #                 print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                #                       .format(epoch+1, num_epochs, global_step, num_epochs*len(),
                #                               average_train_loss, average_valid_loss))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
                else:

                    print("validation loss did not decrease - not saving anything!")

    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


model = LSTM_Model(textData.word2index, textData.index2word, LM=None, i2v=None).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model=model, optimizer=optimizer, num_epochs=5)


train_loss_list, valid_loss_list, global_steps_list = load_metrics(destination_folder + '/metrics.pt')
plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Valid')
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"{destination_folder}training_loss.png")
plt.show()



def evaluate(model, batches=textData.getBatches("test"), threshold=0.5):
    y_pred = []
    y_true = []

    model.eval()
    for index, batch in enumerate(batches):
        x = {}
        x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
        x['enc_len'] = torch.LongTensor(batch.encoder_lens).to(args['device'])
        x['labels'] = autograd.Variable(torch.FloatTensor(batch.label)).to(args['device'])

        output = model(x['enc_input'], x['enc_len'])
        labels = x['labels']

        output = (output > threshold).int()
        y_pred.extend(output.tolist())
        y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['not ra', 'ra'])
    ax.yaxis.set_ticklabels(['not ra', 'ra'])

    plt.savefig(f"{destination_folder}confusion_matrix.png")


best_model = LSTM_Model(textData.word2index, textData.index2word, LM=None, i2v=None).to(device)
optimizer = optim.Adam(best_model.parameters(), lr=0.001)

load_checkpoint(destination_folder + '/model.pt', best_model, optimizer)
evaluate(best_model)



