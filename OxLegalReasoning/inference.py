# import functools
# print = functools.partial(print, flush=True)
from funcsigs import signature
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.init import _calculate_fan_in_and_fan_out
import numpy as np
import datetime, json
import time, math, gzip, random
from Encoder import Encoder
from rcnn_encoder import RCNNEncoder
from Decoder import Decoder
from Hyperparameters import args
from collections import namedtuple
import pandas as pd

import os

from LSTM_IB_GAN_mimic import LSTM_IB_GAN_Model, Discriminator
from textdataMimic import TextDataMimic, Batch
from LanguageModel_mimic import LanguageModel
from tqdm import tqdm
import matplotlib.pyplot as plt


import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report


import argparse
import pandas as pd
from datetime import date


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', '-md')
parser.add_argument('--gpu', '-g')
parser.add_argument('--modelarch', '-m')
parser.add_argument('--choose', '-c')
parser.add_argument('--use_big_emb', '-be')
parser.add_argument('--use_new_emb', '-ne')
parser.add_argument('--date', '-d')
parser.add_argument('--encarch', '-ea')
cmdargs = parser.parse_args()


usegpu = True

if cmdargs.gpu is None:
    usegpu = False
else:
    usegpu = True
    args['device'] = 'cuda:' + str(cmdargs.gpu)

if cmdargs.modelarch is None:
    args['model_arch'] = 'lstmibgan'
else:
    args['model_arch'] = cmdargs.modelarch

if cmdargs.choose is None:
    args['choose'] = 0
else:
    args['choose'] = int(cmdargs.choose)

if cmdargs.use_big_emb:
    args['big_emb'] = True
else:
    args['big_emb'] = False

if cmdargs.use_new_emb:
    args['new_emb'] = True
else:
    args['new_emb'] = False

if cmdargs.date is None:
    args['date'] = str(date.today())

if cmdargs.model_dir is None:
    # args['model_dir'] = "./artifacts/RCNN_IB_GAN_be_mimic3_org_embs2021-05-12.pt"
    args['model_dir'] = "./artifacts/RCNN_IB_GAN_be_mimic3_org_embs_LM2021-05-27.pt"
else:
    args["model_dir"] = "./artifacts/" + str(cmdargs.model_dir)

args['output_dir'] = args['model_dir'][:-3]


if cmdargs.encarch is None:
    args['enc_arch'] = 'rcnn'
else:
    args['enc_arch'] = cmdargs.encarch

# Create output directory if needed
if not os.path.exists(args['output_dir']):
    os.makedirs(args['output_dir'])

full_model = False

do_full_eval = True

get_sentence_results = True

sentence = "has experienced acute on chronic diastolic heart failure in the setting of volume overload due to his sepsis prescribed warfarin due to high sys blood pressure 160 "
# sentence = "High diastolic blood pressure. Wheezy with low blood oxygen levels. Normal resipiratory rate"

textData = TextDataMimic("mimic", "../clinicalBERT/data/", "discharge", trainLM=False,
                         test_phase=False,
                         big_emb=args['big_emb'], new_emb = args["new_emb"])

if args["new_emb"]:
    print("using language model with new 200d embeddings")
    LM = torch.load(args['rootDir'] + '/LMmimic_newembs200.pkl', map_location=args['device'])
else:
    print("using older 100d word embeddings")
    LM = torch.load(args['rootDir'] + '/LMmimic.pkl', map_location=args['device'])
for param in LM.parameters():
    param.requires_grad = False


def main():


    #TODO use below for loading un-trained models to then load state_dicts into
    # org_G_model = LSTM_IB_GAN_mimic.LSTM_IB_GAN_Model(textData.word2index, textData.index2word, LM, textData.index2vector).to(args['device'])
    # # org_G_model = LSTM_IB_GAN_mimic.LSTM_IB_GAN_Model()
    # org_D_model = LSTM_IB_GAN_mimic.Discriminator().to(args['device'])
    # G_optimizer = optim.Adam(G_model.parameters(), lr=0.0004, weight_decay=2e-6)
    # D_optimizer = optim.Adam(D_model.parameters(), lr=0.0004, weight_decay=2e-6)
    # #
    # print("original G model is: ", org_G_model)
    # print("original D model is: ", org_D_model)
    #######################################################################################

    # use below for loading full model

    if full_model:
        print("loading full model")
        model_pretrained = torch.load("./artifacts/LSTM_IB_GAN_be_mimic3_2021-05-04.pt")

        G_model = model_pretrained[0]
        D_model = model_pretrained[1]

        print("the G model is: ", G_model)
        print("=========================")
        print("the D model is : ", D_model)
        output_dir = args["output_dir"]

        # print(G_model.eval())

    else:
    #TODO use below for loading un-trained models to then load state_dicts into
        print("loading model from state dicts")

        checkpoint = torch.load(args['model_dir'])
        G_model = LSTM_IB_GAN_Model(textData.word2index, textData.index2word, LM, textData.index2vector).to(args['device'])
        D_model = Discriminator().to(args['device'])
        G_optimizer = optim.Adam(G_model.parameters(), lr=0.0004, weight_decay=2e-6)
        D_optimizer = optim.Adam(D_model.parameters(), lr=0.0004, weight_decay=2e-6)
        #
        #load model dicts
        G_model.load_state_dict(checkpoint['modelG_state_dict'])
        D_model.load_state_dict(checkpoint['modelD_state_dict'])
        # G_optimizer.load_state_dict(checkpoint['optimizerG_state_dict'])
        # D_optimizer.load_state_dict(checkpoint['optimizerD_state_dict'])

        print("the G model from checkpoint is: ", G_model)
        print("=========================")
        print("the D model from checkpoint is : ", D_model)


        output_dir = args["output_dir"]

    def vote_score(df, score, output_dir):
        print("doing vote score stuff")
        df['pred_score'] = score

        # print(f"df inside of vote score is: {df}")

        # print(len(df))
        # print(len(score))
        df_sort = df.sort_values(by=['ID'])
        # score
        temp = (df_sort.groupby(['ID'])['pred_score'].agg(max) + df_sort.groupby(['ID'])['pred_score'].agg(sum) / 2) / (
                1 + df_sort.groupby(['ID'])['pred_score'].agg(len) / 2)
        x = df_sort.groupby(['ID'])['Label'].agg(np.min).values
        df_out = pd.DataFrame({'logits': temp.values, 'ID': x})

        fpr, tpr, thresholds = roc_curve(x, temp.values)
        auc_score = auc(fpr, tpr)

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Val (area = {:.3f})'.format(auc_score))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')

        plt.savefig(f'{output_dir}/auroc_BioClinicalbert_discharge.png')
        # plt.show()

        #     string = 'auroc_clinicalbert_' + args.readmission_mode + '.png'
        #     plt.savefig(os.path.join(args.output_dir, string))

        return fpr, tpr, df_out


    def pr_curve_plot(y, y_score, output_dir):
        print("making pr_curve_plot")
        precision, recall, _ = precision_recall_curve(y, y_score)
        area = auc(recall, precision)
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})

        plt.figure(2)
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: AUC={0:0.2f}'.format(
            area))

        plt.savefig(f'{output_dir}/auprc_BioClinicalbert_discharge.png')
        # plt.show()


    #     string = 'auprc_clinicalbert_' + args.readmission_mode + '.png'

    #     plt.savefig(os.path.join(args.output_dir, string))


    def vote_pr_curve(df, score, output_dir):
        print("inside vote_pr_curve")
        df['pred_score'] = score
        df_sort = df.sort_values(by=['ID'])
        # score
        temp = (df_sort.groupby(['ID'])['pred_score'].agg(max) + df_sort.groupby(['ID'])['pred_score'].agg(sum) / 2) / (
                1 + df_sort.groupby(['ID'])['pred_score'].agg(len) / 2)
        y = df_sort.groupby(['ID'])['Label'].agg(np.min).values

        precision, recall, thres = precision_recall_curve(y, temp)
        pr_thres = pd.DataFrame(data=list(zip(precision, recall, thres)), columns=['prec', 'recall', 'thres'])
        vote_df = pd.DataFrame(data=list(zip(temp, y)), columns=['score', 'label'])

        pr_curve_plot(y, temp, output_dir)

        temp = pr_thres[pr_thres.prec > 0.799999].reset_index()

        rp80 = 0
        if temp.size == 0:
            print('Test Sample too small or RP80=0')
        else:
            rp80 = temp.iloc[0].recall
            print('Recall at Precision of 80 is {}', rp80)

        return rp80



    def test(textData, model, datasetname, max_accuracy=-1, eps = 1e-6):
        total = 0
        right = 0

        dset = []

        pppt = False
        MSEloss = 0
        total_prec = 0.0
        samplerate = 0.0

        model.eval()
        nb_eval_steps = 0
        eval_acc_total = 0
        TP_c = np.zeros(1)
        FP_c = np.zeros(1)
        FN_c = np.zeros(1)
        TN_c = np.zeros(1)

        all_chosen_words = []
        batch_chosen_words = []
        total_samples_seen = 0

        output_probs_history = []
        output_labels_history = []
        true_labels_history = []


        batch_chosen_words_dfs =[]

        print("device to be used is: ", args['device'])

        data_len = len(textData.datasets[datasetname])
        print("data_len is: ", data_len)


        with torch.no_grad():
            for (i, batch) in tqdm(enumerate(textData.getBatches(datasetname)), total=data_len/args['batchSize']):
                pppt=False
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                # print("on batch ", i)
                x = {}
                x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
                x['enc_len'] = torch.LongTensor(batch.encoder_lens).to(args['device'])
                x['labels'] = autograd.Variable(torch.FloatTensor(batch.label).unsqueeze(1)).to(args['device'])

                # print("original x labels: ", x['labels'])

                # print("enoder inputer is: ", x['enc_input'])
                output_probs, sampled_words = model.predict(x)





                # print("sampled words shape: ", sampled_words.shape)
                # print("output probs: ", output_probs)
                output_labels = np.asarray([1 if i else 0 for i in (output_probs.flatten() >= 0.5)])
                # print("output labels: ", output_labels)

                # use below for  converting from gpu to cpu
                true_labels = x['labels'].long().flatten().cpu().numpy()



                true_labels_history = true_labels_history + true_labels.tolist()
                output_labels_history = output_labels_history + output_labels.tolist()
                output_probs_history  = output_probs_history + output_probs.flatten().tolist()

                output_probs_flat =  output_probs.flatten().cpu().numpy()

                # print(output_probs_flat)
                # print(output_labels)

                # use below if already runinng on cpy
                # true_labels = x['labels'].long().flatten().numpy()


                tmp_eval_accuracy = np.sum(output_labels == true_labels)

                eval_acc_total += tmp_eval_accuracy
                nb_eval_steps += x['labels'].shape[0]

                # words_chosen_from_sample = []
                # words_from_sample = []
                # print("sampled words in full is: ", sampled_words)
                # print("sampled words in full shape is: ", sampled_words.shape)
                # print("sampled words [0] is: ", sampled_words[0])
                # print("sampled words [0] shape is: ", sampled_words[0].shape)
                # print("sampled words [1] is: ", sampled_words[1] )
                # print("sampled words [1] shape is: ", sampled_words[1].shape)
                # print("batch encoderSeqs[0] is: ", batch.encoderSeqs[0])

                for i, b in enumerate(batch.encoder_lens):
                    words_chosen_from_sample = []
                    words_not_chosen_from_sample = []
                    words_from_sample = []
                    batch_chosen_encoder_seqs = []
                    batch_encoder_seqs = []

                    # print(f"working on iteration: {i}")
                    # print(f"the current batch.encoderseqs is: {batch.encoderSeqs[i]}")
                    # print(f"length of current encoder seqs is: {len(batch.encoderSeqs[i])}")
                    # print(f"the current sampled_words is: {sampled_words[i]}")
                    # print(f"length of current sampled words is: {len(sampled_words[i])}")

                    for w, choice in zip(batch.encoderSeqs[i], sampled_words[i]):
                        # print(f"w is : {textData.index2word[w]}")
                        # print(f"choice is: {choice}")
                        # print(f"original sampled words choice [1] is:{sampled_words[1]} ")

                        if choice == 1:
                            # print("This word was selected")

                            words_chosen_from_sample.append(textData.index2word[w])
                            batch_chosen_encoder_seqs.append(w)
                        else:
                            words_not_chosen_from_sample.append(textData.index2word[w])
                            batch_encoder_seqs.append(w)

                        #also append to main list
                        words_from_sample.append(textData.index2word[w])
                    # print(f"Length of words from sample was: {len(words_from_sample)}")
                    # print(f"\n length of chosen words from sample was: {len(words_chosen_from_sample)}")

                    #TODO check if this is going to end up a array containing all test examples with their correpsonding chosen words?
                    # can also try to instead create a dictionary or something
                    all_chosen_words.append(words_chosen_from_sample)
                    check = any(item in words_chosen_from_sample for item in batch.raw[i])



                    # print("original raw sentence words: ", batch.raw[i])
                    #
                    # print("##############################")
                    # print("sampled words for model: ", words_from_sample)
                    # print("##############################")
                    # print("chosen words from sample: ", [words_chosen_from_sample])
                    # print("??????????????????????????????????????")

                    # print("batch.raw is: ", batch.raw[i])
                    # if check is True:
                    #
                    #     if "UNK" not in words_chosen_from_sample:
                    #
                    #         print("wasn't even in there pal")

                    # print("chosen words is: ", all_chosen_words)


                    batch_chosen_words_dfs.append(pd.DataFrame({"raw_sentence": [batch.raw[i]], "chosen_words": [words_chosen_from_sample],
                                                                "words_not_chosen":[words_not_chosen_from_sample],"predicted_label": output_labels[i], "true_label": true_labels[i],
                                                                "proba":output_probs_flat[i],
                                                                "enq_seq_chosen":[batch_chosen_encoder_seqs], "enq_seq_not_chosen":[batch_encoder_seqs]}))

                    # print(all_chosen_words)





                # for w, choice in zip(batch.encoderSeqs[0], sampled_words):
                #     print("word is: ", textData.index2word[w])
                #     print("choice shape is: ", choice.shape)
                #     print("choice [1] is: ", choice[1])
                #
                #     if choice == 1:
                #         print("word was chosen for sample/z_nero: ")
                #         print(f'<***{textData.index2word[w]}***>   \n')
                    # else:
                    #     print("\nthis word wasn't!")
                    #     print(textData.index2word[w], end=' ')









                # for w, choice in zip(batch.encoderSeqs[0], sampled_words):
                #     # print("batch encoderSeqs is: ", batch.encoderSeqs[0])
                #     print("w is: ", w)
                #     # print("w shape is: ", w.shape)
                #     print("choice shape is: ", choice.shape)
                #     print("choice [0] is: ", choice[0])
                #     print("\nchoice [1] is: ", choice[1])
                    #     if choice[1] == 1:
                    #         print("this word was in the sampled words produced by the selector network! ")
                    #         print(f'<***{textData.index2word[w]}***>   \n')
                    #         # logger.info(f'< {textData.index2word[w]} >   ')
                    #     # else:
                    #     #     print("\nthis word wasn't!")
                    #     #     print(textData.index2word[w], end=' ')
                    #     #     # logger.info(f"{textData.index2word[w]}")

                # print("output labls are: ", output_labels)

                # convert output labels and true labels to boolean
                y = np.array(output_labels, dtype = bool)
                answer = np.array(true_labels, dtype = bool)

                # get false positives etc
                tp_c = ((answer == True) & (answer == y)).sum(axis=0)  # c
                fp_c = ((answer == True) & (y == False)).sum(axis=0)  # c
                fn_c = ((answer == False) & (y == True)).sum(axis=0)  # c
                tn_c = ((answer == False) & (y == False)).sum(axis=0)  # c
                TP_c += tp_c
                FP_c += fp_c
                FN_c += fn_c
                TN_c += tn_c


                batch_correct = output_labels == true_labels
                # print(output_labels.size(), torch.LongTensor(batch.label).size())
                right += sum(batch_correct)
                total += x['enc_input'].size()[0]

            # print("printing lenths of output probs and true labels histories")
            # print(len(output_probs_history))
            # print(len(true_labels_history))

            # print(output_probs_history)
            # print(true_labels_history)
            # print(output_labels_history)





            # output_probs_history = np.concatenate(output_probs_history, axis = 0)
            # true_labels_history = np.concatenate(true_labels_history, axis = 0)
            #
            # print("AFTER concat! ++++++ printing lenths of output probs and true labels histories")
            # print(len(output_probs_history))
            # print(len(true_labels_history))

            res = {}

            res['accuracy'] = right / total

            P_c = TP_c / (TP_c + FP_c)
            R_c = TP_c / (TP_c + FN_c)
            F_c = 2 * P_c * R_c / (P_c + R_c)

            res['F_macro_n '] = np.nanmean(F_c)
            res['MP_n'] = np.nanmean(P_c)
            res['MR_n'] = np.nanmean(R_c)

            print("finished testing! ")

            accuracy = res['accuracy']

            print(classification_report(true_labels_history, output_labels_history))
            # save classification report as a dataframe to csv
            pd.DataFrame(classification_report(true_labels_history, output_labels_history, output_dict=True)).to_csv(
                f"{output_dir}/classification_report.csv")

            df_test = pd.read_csv("../clinicalBERT/data/discharge/test.csv")
            print(df_test.shape)

            # print("length of output_probs_history: ", output_probs_history)
            #
            # print(output_probs_history)
            # print(output_labels_history)
            # print(true_labels_history)


            fpr, tpr, df_out = vote_score(df_test, output_probs_history, output_dir)

            rp80 = vote_pr_curve(df_test, output_probs_history, output_dir)

            pd.DataFrame({'rp80':[rp80]}).to_csv(f"{output_dir}/rp80.csv")

            cf = confusion_matrix(true_labels_history, output_labels_history, normalize='true')
            df_cf = pd.DataFrame(cf, ['not r/a', 'readmitted'], ['not r/a', 'readmitted'])
            plt.figure(figsize=(6, 6))
            plt.suptitle("Readmitted vs not readmitted")
            sns.heatmap(df_cf, annot=True, cmap='Blues')
            plt.savefig(f"{output_dir}/Confusion_Matrix.png")

            # print("accuracy based on right/total is: ", accuracy)

            # print("total accuracy is: ", eval_acc_total)
            # print("nb eval steps is: ", nb_eval_steps)
            avg_accuracy = eval_acc_total / nb_eval_steps
            # print("accuracy based on eval_acc_total/nb_eval_steps: ", avg_accuracy)
            print("length of all chosen words: ", len(all_chosen_words))
            #
            # print("saving all chosen words to csv!")
            # pd.DataFrame(all_chosen_words).to_csv("./artifacts/test_all_chosen_words.csv")

            print("saving batch dfs to csv")
            all_batch_chosen_words = pd.concat(batch_chosen_words_dfs)

            print("shpae of all_batch_chosen_words: ", all_batch_chosen_words.shape)
            all_batch_chosen_words.to_csv(f"{output_dir}/batch_all_chosen_words.csv")

        return res, all_chosen_words

    if do_full_eval:
        eval_stats, all_chosen_words = test(textData,G_model,'test', max_accuracy=-1)
        # #
        print("eval_stats: ", eval_stats)



    # input_test =
    # output_probs, sampled_words =  G_model.predict(x)

    def get_one_sample(textData, setname, idx):
        data = textData.datasets[setname][idx]
        # print("data is : ", data)
        # print("len of data is: ", len(data))


        batch = Batch()
        # print("data is : ", data)

        # print("\n batch is: ", batch)

        sen_ids, y, raw_sen, rational = data



        batch.encoderSeqs.append(sen_ids)
        batch.encoder_lens.append(len(batch.encoderSeqs[0]))
        batch.label.append(y)
        batch.rationals.append(rational)
        batch.raw.append(raw_sen)

        # print("updated batch: ", batch)
            # print(y)

        maxlen_enc = max(batch.encoder_lens)
        print("maxlen enc: ", maxlen_enc)


        batch.encoderSeqs[0] = batch.encoderSeqs[0] + [textData.word2index['PAD']] * (
                maxlen_enc - len(batch.encoderSeqs[0]))





        return batch


    def get_metrics_one(model, batch):
        model.eval()
        x = {}
        x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
        x['enc_len'] = torch.LongTensor(batch.encoder_lens).to(args['device'])
        x['labels'] = autograd.Variable(torch.FloatTensor(batch.label).unsqueeze(1)).to(args['device'])
        output_probs, sampled_words = model.predict(x)



        print("sampled words: ", sampled_words)
        print("sampled words shape: ", sampled_words.shape)
        # print("sampled words shape: ", sampled_words.shape)
        # print("output probs: ", output_probs)
        output_labels = np.asarray([1 if i else 0 for i in (output_probs.flatten() >= 0.5)])
        # print("output labels: ", output_labels)

        # use below for  converting from gpu to cpu
        true_labels = x['labels'].long().flatten().cpu().numpy()
        # use below if already runinng on cpy
        # true_labels = x['labels'].long().flatten().numpy()
        print("zero index batch encoder seqs shape is  is ", len(batch.encoderSeqs[0]))

        all_chosen_words = []
        batch_chosen_words_dfs = []
        words_from_sample = []
        for w, choice in zip(batch.encoderSeqs[0], sampled_words[0]):
            # print("word is: ", textData.index2word[w])
            # print("choice is: ", choice)
            if choice ==1:
                print("word was chosen for sample/z_nero: ")
                print(f'<***{textData.index2word[w]}***>   \n')
                all_chosen_words.append(textData.index2word[w])
            # else:
            #     print("\nthis word wasn't!")
            #     print(textData.index2word[w], end=' ')

            words_from_sample.append(textData.index2word[w])

            batch_chosen_words_dfs = pd.DataFrame({"raw_sentence":batch.raw,"chosen_words":[all_chosen_words]})
        print(batch_chosen_words_dfs)


        # print("original raw sentence words: ", batch.raw)
        # print(len(batch.raw))
        # print("##############################")
        # print("sampled words for model: ", words_from_sample)
        # print("##############################")
        # print("chosen words from sample: ", [all_chosen_words])
        # print(len(all_chosen_words ))

        batch_chosen_words_dfs.to_csv(f"{output_dir}/test_sample_chosen_words.csv")


        # for w, choice in zip(batch.encoderSeqs[0], sampled_words):
        #
        #     print("batch encoderSeqs is: ", batch.encoderSeqs[0])
        #     print("w is: ", w)
        #     print("choice shape is: ", choice.shape)
        #     print("choice [0] is: ", choice[0])
        #     print("\nchoice [1] is: ", choice[1])
        #
        #
        #     if choice[1] == 1:
        #         print("this word was in the sampled words produced by the selector network! ")
        #         print(f'<***{textData.index2word[w]}***>   \n')
        #         # logger.info(f'< {textData.index2word[w]} >   ')
        #     else:
        #         print("\nthis word wasn't!")
        #         print(textData.index2word[w], end=' ')
                # logger.info(f"{textData.index2word[w]}")
        #
    #
    #
    # test_batch = get_one_sample(textData,"test", 83)
    # test_results = get_metrics_one(G_model,test_batch)


    def sentence2results(model, sentence):
        print(f"getting chosen words for following sentence \n {sentence}")
        model.eval()
        batch = textData.sentence2batch(sentence)


        x = {}
        x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
        x['enc_len'] = torch.LongTensor(batch.encoder_lens).to(args['device'])
        x['labels'] = autograd.Variable(torch.FloatTensor(batch.label)).to(args['device'])

        print(x["labels"])
        output_probs, sampled_words = model.predict(x)

        output_labels = np.asarray([1 if i else 0 for i in (output_probs.flatten() >= 0.5)])

        output_probs_flat = output_probs.flatten().cpu().detach().numpy()

        all_chosen_words = []
        all_not_chosen_words = []
        chosen_enc_seqs = []
        not_chosen_enc_seqs = []

        words_from_sample = []
        for w, choice in zip(batch.encoderSeqs[0], sampled_words[0]):
            # print("word is: ", textData.index2word[w])
            # print("choice is: ", choice)
            if choice == 1:
                print("word was chosen for sample/z_nero: ")
                print(f'<***{textData.index2word[w]}***>   \n')
                all_chosen_words.append(textData.index2word[w])
                chosen_enc_seqs.append(w)
            else:
            #     print("\nthis word wasn't!")
            #     print(textData.index2word[w], end=' ')
                all_not_chosen_words.append(textData.index2word[w])
                not_chosen_enc_seqs.append(w)

            words_from_sample.append(textData.index2word[w])



        chosen_words_df =  pd.DataFrame({"raw_sentence": batch.raw, "chosen_words": [all_chosen_words],
                      "words_not_chosen": [all_not_chosen_words], "predicted_label": output_labels,"proba":output_probs_flat,
                      "enq_seq_chosen": [chosen_enc_seqs], "enq_seq_not_chosen": [not_chosen_enc_seqs]})

        # print(batch_chosen_words_dfs)

        # print("original raw sentence words: ", batch.raw)
        # print(len(batch.raw))
        # print("##############################")
        # print("sampled words for model: ", words_from_sample)
        # print("##############################")
        # print("chosen words from sample: ", [all_chosen_words])
        # print(len(all_chosen_words ))

        chosen_words_df.to_csv(f"{output_dir}/test_sentence_chosen_words.csv")

    if get_sentence_results:

        sentence2results(G_model, sentence)


if __name__ == "__main__":
    main()
