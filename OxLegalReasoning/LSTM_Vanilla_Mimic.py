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
import torch.nn.functional as F

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
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, \
    ExponentialLR
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




textData = TextDataMimic("mimic", "../clinicalBERT/data/", "discharge", trainLM=False, test_phase=False, big_emb = False, new_emb = False)
        # self.start_token = self.textData.word2index['START_TOKEN']
        # self.end_token = self.textData.word2index['END_TOKEN']
args['vocabularySize'] = textData.getVocabularySize()

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
        self.index2vector = i2v
        self.max_length = 1000



        # TODO try using the language model embedding after training!
        #         self.embedding = LM.embedding
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(i2v))
        self.emb_size = self.index2vector.shape[1]
        print(f"self embedding size is: ", self.emb_size)

        # print(self.embedding)

        # self.embedding = nn.Embedding(textData.getVocabularySize(), 200)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=self.emb_size,
                            hidden_size=self.dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.1)

        # self.max_pool = torch.max()
        self.fc_max = nn.Linear(1,50)
        self.fc_max2 = nn.Linear(50,1)
        self.fc = nn.Linear(2 * dimension, 1)




    def forward(self, text, text_len):
        #         self.encoderInputs = encoderInputs.to(args['device'])
        #         self.encoder_lengths = encoder_lengths

        text_emb = self.embedding(text)
        text_len = text_len.cpu()

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output,  (h_n, c_n) = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # print(f"output shape is: {output.shape}")
        # print(f"h_n shape is : {h_n.shape}" )

        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        # print("out reduced shape is: ", out_reduced.shape)
        text_fea = self.drop(out_reduced)
        # print(f"text feature after concat is: ", text_fea)
        # print("concated output shape: ", out_reduced.shape)
        max_pooled,_ = torch.max(text_fea,1)

        max_pooled = torch.unsqueeze(max_pooled,1)
        # print(f"max pooled shape is: {max_pooled.shape}")
        max_out1 = self.fc_max(max_pooled)
        # print(f"max out1 shape before rect is: ", max_out1.shape)
        max_out1 = F.relu(max_out1)
        # print(f"max_out1 after relu is: ", max_out1)
        # print(f"max out1 shape after relu is: ", max_out1.shape)
        max_out_final = self.fc_max2(max_out1)
        # print(f"max final shape is: ", max_out_final.shape)



        # print(max_pooled)


        #use this for max pooling
        text_out = torch.squeeze(max_out_final,1)
        text_out = torch.sigmoid(text_out)


        #use this if not using max pool
        # text_fea = self.fc(text_fea)
        # text_fea = torch.squeeze(text_fea, 1)
        # text_out = torch.sigmoid(text_fea)
        # print(f"text out is: {text_out}")

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


def save_metrics(save_path, train_loss_list, valid_loss_list, epoch_num_list):
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'epoch_num_list': epoch_num_list}

    torch.save(state_dict, save_path)
    print(f'Model metrics saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model metrics loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['epoch_num_list']


# define metric
def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(preds)

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


# Training Function
destination_folder = "./results/LSTM_Vanilla_strict/"
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
device = args["device"]
args["batchSize"] = 128
eval_every = 50


def train(model,
          optimizer,
          criterion=nn.BCELoss(),
          num_epochs=50,
          file_path=destination_folder,
          best_valid_loss=float("Inf")):
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    epoch_loss_history = []
    epoch_val_loss_history = []
    global_steps_list = []

    batches = textData.getBatches()
    total_steps = num_epochs * len(batches)

    # training loop
    model.train()
    for epoch in range(num_epochs):
        train_loss_list = []
        val_loss_list = []
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

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
            train_loss_list.append(loss.item())

        # run validation at end of epoch
        model.eval()
        with torch.no_grad():
            val_batches = textData.getBatches("dev")

            #                   # validation loop
            for index, batch in enumerate(val_batches):
                x = {}
                x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
                x['enc_len'] = torch.LongTensor(batch.encoder_lens).to(args['device'])
                x['labels'] = autograd.Variable(torch.FloatTensor(batch.label)).to(args['device'])

                output = model(x['enc_input'], x['enc_len'])
                loss = criterion(output, x['labels'])
                valid_running_loss += loss.item()
                val_loss_list.append(loss.item())

        scheduler.step()
        # get the epoch train and val losses
        average_train_loss = running_loss / len(batches)
        average_valid_loss = valid_running_loss / len(val_batches)

        epoch_loss = (sum(train_loss_list)) / len(train_loss_list)
        #         print(f"epoch training loss: {epoch_loss}")
        epoch_loss_history.append(epoch_loss)

        epoch_val_loss = (sum(val_loss_list)) / len(val_loss_list)
        epoch_val_loss_history.append(epoch_val_loss)
        global_steps_list.append(global_step)

        #        # resetting running values
        running_loss = 0.0
        valid_running_loss = 0.0
        model.train()

        # print progress

        print(
            f"Epoch number: {epoch}. Epoch loss: {epoch_loss}\n Val loss: {average_valid_loss}\n Step {global_step}/{total_steps}")

        # checkpoint
        if best_valid_loss > average_valid_loss:
            print("valid loss decreased - saving model")
            best_valid_loss = average_valid_loss
            epoch_number_list = list(range(len(epoch_loss_history)))
            save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
            save_metrics(file_path + '/metrics.pt', train_loss_list, val_loss_list, epoch_number_list)
        else:
            print("valid loss did not decrease - not saving!")

    epoch_number_list = list(range(len(epoch_loss_history)))
    save_metrics(file_path + '/metrics.pt', epoch_loss_history, epoch_val_loss_history, epoch_number_list)
    print('Finished Training!')


model = LSTM_Model(textData.word2index, textData.index2word, LM=None, i2v=textData.index2vector).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=4e-3)
#standard step scheduler
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#multistep
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,20,50], gamma=0.1)

# scheduler = ReduceLROnPlateau(
#             optimizer, mode='min', factor=args["lr_decay"],
#             patience=args["patience"],
#             threshold=args["threshold"], threshold_mode='rel',
#             cooldown=args["cooldown"], verbose=True, min_lr=args["min_lr"])

train(model=model, optimizer=optimizer, num_epochs=10)


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
    output_probs_history = []


    model.eval()
    for index, batch in enumerate(batches):
        x = {}
        x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
        x['enc_len'] = torch.LongTensor(batch.encoder_lens).to(args['device'])
        x['labels'] = autograd.Variable(torch.FloatTensor(batch.label)).to(args['device'])

        output_probs = model(x['enc_input'], x['enc_len'])

        # print(f"output probs: {output_probs}")
        output_probs_history.extend(output_probs.tolist())
        labels = x['labels']

        output_labels = (output_probs > threshold).int()

        # print(f"output labels: {output_labels}")

        y_pred.extend(output_labels.tolist())
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
    plt.show()


    eval_probs = {}
    eval_probs["proba"] = output_probs_history
    eval_probs["pred_label"] = y_pred

    pd.DataFrame(eval_probs).to_csv(f"{destination_folder}lstm_vanilla_eval_probs.csv", index = False)


best_model = LSTM_Model(textData.word2index, textData.index2word, LM=None, i2v=textData.index2vector).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

load_checkpoint(destination_folder + '/model.pt', best_model, optimizer)
evaluate(best_model)



