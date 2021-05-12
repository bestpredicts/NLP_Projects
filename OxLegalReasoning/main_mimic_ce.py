# Copyright 2020 . All Rights Reserved.
# Author : Lei Sha

from Hyperparameters import args
import argparse
from datetime import date

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
parser.add_argument('--modelarch', '-m')
parser.add_argument('--aspect', '-a')
parser.add_argument('--choose', '-c')
parser.add_argument('--use_big_emb', '-be')
parser.add_argument('--date', '-d')
cmdargs = parser.parse_args()
print(cmdargs)
usegpu = True
if cmdargs.gpu is None:
    usegpu = False

    args['device'] = 'cpu'
else:
    usegpu = True
    args['device'] = 'cuda:' + str(cmdargs.gpu)

if cmdargs.modelarch is None:
    args['model_arch'] = 'lstm'
else:
    args['model_arch'] = cmdargs.modelarch

if cmdargs.aspect is None:
    args['aspect'] = 0
else:
    args['aspect'] = int(cmdargs.aspect)
if cmdargs.choose is None:
    args['choose'] = 0
else:
    args['choose'] = int(cmdargs.aspect)
if cmdargs.use_big_emb:
    args['big_emb'] = True
else:
    args['big_emb'] = False
if cmdargs.date is None:
    args['date'] = str(date.today())

#set number of labels

args['num_classes'] = 2

import functools
print = functools.partial(print, flush=True)
import os

from textdataMimic import TextDataMimic
import time, sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time, datetime
import math, random
import nltk
import pickle
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# import matplotlib.pyplot as plt
import numpy as np
import copy
from LanguageModel_mimic import LanguageModel

import LSTM_IB_GAN_mimic
import LSTM_IB_GAN_mimic_ce


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (%s)' % (asMinutes(s), datetime.datetime.now())


class Runner:
    def __init__(self):
        self.model_path = args['rootDir'] + '/model_mimic_ce_030521' + args['model_arch'] + args['date'] + '.pt'

    def main(self):

        if args['model_arch'] in ['lstmibgan']:
            args['classify_type'] = 'single'
            args['batchSize'] = 256

        self.textData = TextDataMimic("mimic", "../clinicalBERT/data/", "discharge", trainLM=False, test_phase=True, big_emb = args['big_emb'])
        # self.start_token = self.textData.word2index['START_TOKEN']
        # self.end_token = self.textData.word2index['END_TOKEN']
        args['vocabularySize'] = self.textData.getVocabularySize()
        args['chargenum'] = 2
        args['embeddingSize'] = self.textData.index2vector.shape[1]
        print(self.textData.getVocabularySize())
        args['model_arch'] = 'lstmibgan'
        # args['aspect'] = 0
        args['hiddenSize'] = 200

        print(args)
        if args['model_arch'] == 'lstmibgan':
            print('Using LSTM information bottleneck GAN model for mimic.')
            LM = torch.load(args['rootDir']+'/LMmimic.pkl', map_location=args['device'])
            for param in LM.parameters():
                param.requires_grad = False

            # ppl = self.CalPPL(LM)
            # print('PPL=',ppl)
            # LM=0
            LSTM_IB_GAN_mimic_ce.train(self.textData, LM, self.textData.index2vector)

    #TODO adapt below function - its taking from main_small...
    def test(self, datasetname, max_accuracy, eps=1e-20):
        # if not hasattr(self, 'testbatches'):
        #     self.testbatches = {}
        # if datasetname not in self.testbatches:
        # self.testbatches[datasetname] = self.textData.getBatches(datasetname)
        right = 0
        total = 0

        dset = []

        exact_match = 0
        p = 0.0
        r = 0.0
        acc = 0.0

        TP_c = np.zeros(args['num_classes'])
        FP_c = np.zeros(args['num_classes'])
        FN_c = np.zeros(args['num_classes'])
        TN_c = np.zeros(args['num_classes'])

        with torch.no_grad():
            pppt = False
            for batch in self.textData.getBatches(datasetname):
                x = {}
                x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
                x['enc_len'] = batch.encoder_lens
                x['labels'] = autograd.Variable(torch.LongTensor(batch.label)).to(args['device'])

                if args['model_arch'] in ['lstmiterib', 'lstmgrid', 'lstmgmib']:
                    answer = self.model.predict(x).cpu().numpy()
                    #TODO investigate the below line - no idea what this +2 business is
                    y = F.one_hot(torch.LongTensor(batch.label), num_classes=args['num_classes'] + 2)
                    y = y[:, :, :args['num_classes']]  # add content class
                    y, _ = torch.max(y, dim=1)
                    y = y.bool().numpy()
                    exact_match += ((answer == y).sum(axis = 1) == args['num_classes']).sum()
                    total += answer.shape[0]
                    tp_c = ((answer == True) & (answer == y)).sum(axis = 0) # c
                    fp_c = ((answer == True) & (y == False)).sum(axis = 0) # c
                    fn_c = ((answer == False) & (y == True)).sum(axis = 0) # c
                    tn_c = ((answer == False) & (y == False)).sum(axis = 0) # c
                    TP_c += tp_c
                    FP_c += fp_c
                    FN_c += fn_c
                    TN_c += tn_c
                    right = exact_match
                else:
                    output_probs, output_labels = self.model.predict(x)
                    if args['model_arch'] == 'lstmib' or args['model_arch'] == 'lstmibcp':
                        output_labels, sampled_words, wordsamplerate = output_labels
                        if not pppt:
                            pppt = True
                            for w, choice in zip(batch.encoderSeqs[0], sampled_words[0]):
                                if choice[1] == 1:
                                    print(self.textData.index2word[w], end='')
                            print('sample rate: ', wordsamplerate[0])
                    elif args['model_arch'] == 'lstmcapib':
                        output_labels, sampled_words, wordsamplerate = output_labels
                        if not pppt:
                            pppt = True
                            for w, choice in zip(batch.encoderSeqs[0], sampled_words[0, output_labels[0], :]):
                                if choice == 1:
                                    print(self.textData.index2word[w], end='')
                            print('sample rate: ', wordsamplerate[0])

                    batch_correct = output_labels.cpu().numpy() == torch.LongTensor(batch.label).cpu().numpy()
                    right += sum(batch_correct)
                    total += x['enc_input'].size()[0]

                    for ind, c in enumerate(batch_correct):
                        if not c:
                            dset.append((batch.encoderSeqs[ind], batch.label[ind], output_labels[ind]))

        accuracy = right / total

        if accuracy > max_accuracy:
            with open(args['rootDir'] + '/error_case_' + args['model_arch'] + '.txt', 'w') as wh:
                for d in dset:
                    wh.write(''.join([self.textData.index2word[wid] for wid in d[0]]))
                    wh.write('\t')
                    # wh.write(self.textData.lawinfo['i2c'][int(d[1])])
                    wh.write('\t')
                    # wh.write(self.textData.lawinfo['i2c'][int(d[2])])
                    wh.write('\n')
            wh.close()
        if args['model_arch'] in ['lstmiterib', 'lstmgrid', 'lstmgmib']:
            P_c = TP_c / (TP_c + FP_c)
            R_c = TP_c / (TP_c + FN_c)
            F_c = 2 * P_c * R_c / (P_c + R_c)
            F_macro = np.nanmean(F_c)
            TP_micro = np.sum(TP_c)
            FP_micro = np.sum(FP_c)
            FN_micro = np.sum(FN_c)

            P_micro = TP_micro / (TP_micro + FP_micro)
            R_micro = TP_micro / (TP_micro + FN_micro)
            F_micro = 2 * P_micro * R_micro / (P_micro + R_micro)
            S = 100 * (F_macro + F_micro) / 2
            return accuracy, exact_match / total, p, r, acc, F_macro, F_micro, S
        else:
            return accuracy

    def indexesFromSentence(self, sentence):
        return [self.textData.word2index[word] if word in self.textData.word2index else self.textData.word2index['UNK']
                for word in sentence]

    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        # indexes.append(self.textData.word2index['END_TOKEN'])
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def evaluate(self, sentence, correctlabel, max_length=20):
        with torch.no_grad():
            input_tensor = self.tensorFromSentence(sentence)
            input_length = input_tensor.size()[0]
            # encoder_hidden = encoder.initHidden()

            # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            x = {}
            # print(input_tensor)
            x['enc_input'] = torch.transpose(input_tensor, 0, 1)
            x['enc_len'] = [input_length]
            x['labels'] = [correctlabel]
            # print(x['enc_input'], x['enc_len'])
            # print(x['enc_input'].shape)
            decoded_words, label, _ = self.model.predict(x, True)

            return decoded_words, label

    def evaluateRandomly(self, n=10):
        for i in range(n):
            sample = random.choice(self.textData.datasets['train'])
            print('>', sample)
            output_words, label = self.evaluate(sample[2], sample[1])
            output_sentence = ' '.join(output_words[0])  # batch=1
            print('<', output_sentence, label)
            print('')

    def CalPPL(self, LM):

        batches = self.textData.getBatches('dev')
        total = 0
        loss_sum = 0
        for index, batch in enumerate(batches):
            x = {}
            print("supposed batch: ", batch.decoderSeqs)
            x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs)).to(args['device'])
            x['dec_len'] = batch.decoder_lens
            x['dec_target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs)).to(args['device'])
            total += x['dec_input'].size()[0]
            print(x['dec_input'].size())
            print(x)

            embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.textData.index2vector))
            decoderTargetsEmbeddings = embedding(x['dec_target'])
            print("decoder target embeddings shape: ", decoderTargetsEmbeddings.shape)
            _, recon_loss = LM.getloss(x['dec_input'],decoderTargetsEmbeddings, x['dec_target']  )
            loss_sum += recon_loss.sum()

        loss_mean = loss_sum / total
        return torch.exp(loss_mean)


if __name__ == '__main__':

    r = Runner()
    r.main()