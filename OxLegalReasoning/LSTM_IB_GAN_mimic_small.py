import functools
print = functools.partial(print, flush=True)
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

import numpy as np
import datetime
import time, math
from Encoder import Encoder
from Decoder import Decoder
from Hyperparameters import args
import argparse
import pandas as pd
from datetime import date

import argparse
import pandas as pd
from datetime import date


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
parser.add_argument('--modelarch', '-m')
parser.add_argument('--choose', '-c')
parser.add_argument('--use_big_emb', '-be')
parser.add_argument('--use_new_emb', '-ne')
parser.add_argument('--date', '-d')
parser.add_argument('--model_dir', '-md')
parser.add_argument('--encarch', '-ea')
cmdargs = parser.parse_args()

usegpu = True

if cmdargs.gpu is None:
    usegpu = False
else:
    usegpu = True
    args['device'] = 'cuda:' + str(cmdargs.gpu)

if cmdargs.modelarch is None:
    args['model_arch'] = 'lstm'
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
    emb_file_path = "newemb200"
else:
    args['new_emb'] = False
    emb_file_path =  "orgembs"

if cmdargs.date is None:
    args['date'] = str(date.today())

if cmdargs.model_dir is None:
    # args['model_dir'] = "./artifacts/RCNN_IB_GAN_be_mimic3_org_embs2021-05-12.pt"
    args['model_dir'] = "./artifacts/RCNN_IB_GAN_be_mimic3_org_embs_LM2021-05-25.pt"
else:
    args["model_dir"] = str(cmdargs.model_dir)

if cmdargs.encarch is None:
    args['enc_arch'] = 'rcnn'
else:
    args['enc_arch'] = cmdargs.encarch


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), datetime.datetime.now())


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        print('Discriminator creation...')
        self.NLLloss = torch.nn.NLLLoss(reduction='none')
        self.disc = nn.Sequential(
            nn.Linear(args['hiddenSize']*2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid(),
        ).to(args['device'])

    def forward(self, z_nero):
        '''
        :param unsampled_word: batch seq hid
        :return:
        '''

        G_judge_score = self.disc(z_nero)
        return G_judge_score


class LSTM_IB_GAN_Model(nn.Module):
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, w2i, i2w, LM, i2v):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(LSTM_IB_GAN_Model, self).__init__()
        print("Model creation...")

        self.LM = LM
        self.word2index = w2i
        self.index2word = i2w
        self.index2vector = i2v
        self.max_length = args['maxLengthDeco']

        self.NLLloss = torch.nn.NLLLoss(reduction='none')
        self.CEloss = torch.nn.CrossEntropyLoss(reduction='none')

        #TODO this uses this embedding - but need to retrain the language model really or just use the pretrained embeddings?
        # self.embedding = LM.embedding
        # print("using pretrained embeddings yo")
        self.embedding = LM.embedding
        # self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(i2v))
        self.embedding.weight.requires_grad = True

        self.encoder_all = Encoder(w2i, i2w, self.embedding, bidirectional = True).to(args['device'])
        self.encoder_select = Encoder(w2i, i2w, self.embedding, bidirectional = True).to(args['device'])
        self.encoder_mask = Encoder(w2i, i2w, self.embedding, bidirectional = True).to(args['device'])

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.x_2_prob_z = nn.Sequential(
            nn.Linear(args['hiddenSize'] * 2, 2)
        ).to(args['device'])
        self.z_to_fea = nn.Linear(args['hiddenSize']*2, args['hiddenSize']*2).to(args['device'])

        self.ChargeClassifier = nn.Sequential(
            nn.Linear(args['hiddenSize']*2, args['chargenum']),
            nn.LogSoftmax(dim=-1)
        ).to(args['device'])

        self.attm = Parameter(torch.rand(args['hiddenSize']*2, args['hiddenSize'] * 4)).to(args['device'])

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(args['device'])
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature=args['temperature']):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard, y

    def build(self, x, eps=0.000001):
        '''
        :param encoderInputs: [batch, enc_len]
        :param decoderInputs: [batch, dec_len]
        :param decoderTargets: [batch, dec_len]
        :return:
        '''

        # print(x['enc_input'])
        self.encoderInputs = x['enc_input'].to(args['device'])
        self.encoder_lengths = x['enc_len']
        self.classifyLabels = x['labels'].to(args['device'])
        self.batch_size = self.encoderInputs.size()[0]
        self.seqlen = self.encoderInputs.size()[1]

        mask = torch.sign(self.encoderInputs).float()

        # print("encoder inputs during build: ", self.encoderInputs)
        # print("encoder lengths during build: ", self.encoder_lengths)
        # print("mask during build is: ", mask)
        en_outputs, en_state = self.encoder_all(self.encoderInputs, self.encoder_lengths)  # batch seq hid

        # print("encoder outputs : ", en_outputs)
        # print("encoder state : ", en_state)

        en_hidden, en_cell = en_state  # 2 batch hid
        # print(en_hidden.size())
        en_hidden = en_hidden.transpose(0, 1)
        en_hidden = en_hidden.reshape(self.batch_size, args['hiddenSize'] * 4)
        att1 = torch.einsum('bsh,hg->bsg', en_outputs, self.attm)
        att2 = torch.einsum('bsg,bg->bs', att1, en_hidden)
        att2 = self.softmax(att2)
        z_nero_best = torch.einsum('bsh,bs->bh',en_outputs , att2)

        # z_nero_best = self.z_to_fea(en_outputs)
        # z_nero_best, _ = torch.max(z_nero_best, dim=1)  # batch hid
        # print(z_nero_best.size())
        output_all = self.ChargeClassifier(z_nero_best).to(args['device'])  # batch chargenum
        recon_loss_all = self.NLLloss(output_all, self.classifyLabels).to(args['device'])
        recon_loss_mean_all = recon_loss_all #torch.mean(recon_loss_all, 1).to(args['device'])
        # try:
        en_outputs_select, en_state = self.encoder_select(self.encoderInputs, self.encoder_lengths)  # batch seq hid
        # except:
        #     print(self.encoderInputs, self.encoderInputs.size(), self.encoder_lengths)
        #     en_outputs_select, en_state = self.encoder_select(self.encoderInputs, self.encoder_lengths)  # batch seq hid
        # print(en_outputs.size())
        z_logit = self.x_2_prob_z(en_outputs_select.to(args['device']))  # batch seq 2

        z_logit_fla = z_logit.reshape((self.batch_size * self.seqlen, 2))
        sampled_seq, sampled_seq_soft = self.gumbel_softmax(z_logit_fla) # batch seq  //0-1
        sampled_seq = sampled_seq.reshape((self.batch_size, self.seqlen, 2))
        sampled_seq_soft = sampled_seq_soft.reshape((self.batch_size, self.seqlen, 2))
        sampled_seq = sampled_seq * mask.unsqueeze(2)
        sampled_seq_soft = sampled_seq_soft * mask.unsqueeze(2)
        # print(sampled_seq)

        # sampled_word = self.encoderInputs * (sampled_seq[:,:,1])  # batch seq

        en_outputs_masked, en_state = self.encoder_mask(self.encoderInputs, self.encoder_lengths,
                                                        sampled_seq[:, :, 1])  # batch seq hid
        # print("about to apply z_to_fea to the following: ", en_outputs_masked)
        s_w_feature = self.z_to_fea(en_outputs_masked)
        # print("swf aka z_to_fea is: ", s_w_feature)
        z_nero_sampled, _ = torch.max(s_w_feature, dim=1)  # batch hid

        z_prob = self.softmax(z_logit)
        # I_x_z = torch.mean(-torch.log(z_prob[:, :, 0] + eps), 1)
        I_x_z = (z_prob * torch.log(z_prob / torch.FloatTensor([0.9999,0.0001]).unsqueeze(0).unsqueeze(1).to(args['device']))+eps).sum(2).sum(1) * 0.01

        logp_z0 = torch.log(z_prob[:, :, 0])  # [B,T], log P(z = 0 | x)
        logp_z1 = torch.log(z_prob[:, :, 1])  # [B,T], log P(z = 1 | x)
        logpz = torch.where(sampled_seq[:, :, 1] == 0, logp_z0, logp_z1)
        logpz = mask * logpz

        # print(I_x_z)
        # en_hidden, en_cell = en_state   #2 batch hid

        #TODO below omega works
        # omega = torch.mean(torch.sum(torch.abs(sampled_seq[:,:-1,1] - sampled_seq[:,1:,1]), dim = 1))
        #TODO below omega calc does not work
        # omega = self.LM.LMloss(sampled_seq_soft[:,:,1],sampled_seq[:, :, 1], self.encoderInputs)
        # below is from the other ibgan model - based on beer data
        omega = self.LM.LMloss(sampled_seq[:, :, 1], self.encoderInputs)

        # print("omega is: ", omega)
        # print(I_x_z.size(), omega.size())
        # omega = torch.mean(omega, 1)

        output = self.ChargeClassifier(z_nero_sampled).to(args['device'])  # batch chargenum
        recon_loss = self.NLLloss(output, self.classifyLabels).to(args['device'])
        recon_loss_mean = recon_loss#  torch.mean(recon_loss, 1).to(args['device'])

        tt = torch.stack([recon_loss_mean.mean(), recon_loss_mean_all.mean(), I_x_z.mean(), omega.mean()])
        wordnum = torch.sum(mask, dim=1)
        sampled_num = torch.sum(sampled_seq[:,:,1], dim = 1) # batch
        sampled_num = (sampled_num == 0).float()  + sampled_num
        optional = {}
        optional["recon_loss"] = recon_loss_mean.mean().item()  # [1]
        optional['recon_best'] = recon_loss_mean_all.mean().item()
        optional['I_x_z'] = I_x_z.mean().item()
        optional['zdiff'] = omega.mean().item()
        # try:
        #     print("trying to run get_z_stats")
        #     num_0, num_c, num_1, total = self.get_z_stats(sampled_seq, mask)
        #     optional["p0"] = num_0 / float(total)
        #     optional["pc"] = num_c / float(total)
        #     optional["p1"] = num_1 / float(total)
        #     optional["p>0.5"] = (z_prob[:, :, 1] * mask > 0.5).sum().item() / float(total)
        #     optional["p<0.5"] = (z_prob[:, :, 0] * mask > 0.5).sum().item() / float(total)
        #     optional["selected"] = optional["p1"] + optional["pc"]
        # except:
        #     print(optional)
        #     exit(0)

        return recon_loss_mean , recon_loss_mean_all , 0.0003 * I_x_z , 0.005*omega, z_nero_best, z_nero_sampled, output, sampled_seq, sampled_num/wordnum, logpz, optional

    def get_z_stats(self, z=None, mask=None, eps=1e-6):
        """
        Computes statistics about how many zs are
        exactly 0, continuous (between 0 and 1), or exactly 1.
        :param z:
        :param mask: mask in [B, T]
        :return:
        """

        z = torch.where(mask > 0, z, z.new_full([1], 1e2))

        num_0 = (z < eps).sum().item()
        num_c = ((eps < z) & (z < 1. - eps)).sum().item()
        num_1 = ((z > 1. - eps) & (z < 1 + eps)).sum().item()

        total = num_0 + num_c + num_1
        mask_total = mask.sum().item()
        try:
            assert total == mask_total, "total mismatch"
        except:
            print(z, mask)
            print(num_0, num_1, num_c, total, mask_total)
            assert total == mask_total, "total mismatch"
        return num_0, num_c, num_1, mask_total

    def forward(self, x):

        losses,losses_best,I, om, z_nero_best, z_nero_sampled, _, _,_,logpz,optional = self.build(x)
        return losses,losses_best,I, om, z_nero_best, z_nero_sampled, logpz, optional

    def predict(self, x):
        _, _,_,_, _, _,output,sampled_words, wordsamplerate, _, _ = self.build(x)
        return output, (torch.argmax(output, dim=-1), sampled_words, wordsamplerate)


def train(textData, LM, model_path=args['rootDir'] + '/' + args['enc_arch'] + 'SMALL_IB_GAN_be_mimic3_' + emb_file_path+ '_LM' + args['date'] + '.pt', print_every=40, plot_every=10,
          learning_rate=0.001, n_critic=5, eps = 1e-6):
    print('Using small arch...')
    start = time.time()
    plot_losses = []
    print_Gloss_total = 0  # Reset every print_every
    plot_Gloss_total = 0  # Reset every plot_every
    print_Dloss_total = 0  # Reset every print_every
    plot_Dloss_total = 0  # Reset every plot_every
    G_model = LSTM_IB_GAN_Model(textData.word2index, textData.index2word, LM, textData.index2vector).to(args['device'])
    D_model = Discriminator().to(args['device'])

    print(type(textData.word2index))

    G_optimizer = optim.Adam(G_model.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)
    D_optimizer = optim.Adam(D_model.parameters(), lr=learning_rate, eps=1e-3, amsgrad=True)

    iter = 1
    batches = textData.getBatches()
    n_iters = len(batches)
    print('niters ', n_iters)

    args['trainseq2seq'] = False

    max_accu = -1

    # create empty lists for running metrics
    val_accuracy_history = []
    Gloss_history = []
    Dloss_history = []
    F_macro_history = []
    MP_history = []
    MR_history = []
    F_macro_n_history = []
    MP_n_history = []
    MR_n_history = []


    training_stats={}

    # accuracy = test(textData, G_model, 'test', max_accu)
    for epoch in range(args['numEpochs']):
        Glosses = []
        Dlosses = []

        for index, batch in enumerate(batches):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # for param in G_model.parameters():
            #     param.requires_grad = False
            # for param in D_model.parameters():
            #     param.requires_grad = True

            # for ind in range(index, index+5):
            #     ind = ind % n_iters
            D_optimizer.zero_grad()
            x = {}
            x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs)).to(args['device'])
            x['enc_len'] = batch.encoder_lens
            x['labels'] = autograd.Variable(torch.LongTensor(batch.label)).to(args['device'])
            if args['model_arch'] in ['lstmibgan', 'lstmibgan_law']:
                x['labels'] = x['labels']

            # print("current X going into G_model : ", x)
            Gloss_pure,Gloss_best, I, om, z_nero_best, z_nero_sampled, logpz, optional = G_model(x)  # batch seq_len outsize
            Dloss = -torch.mean(torch.log(D_model(z_nero_best).clamp(eps,1))) + torch.mean(torch.log(D_model(z_nero_sampled.detach()).clamp(eps,1)))

            Dloss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(D_model.parameters(), args['clip'])

            D_optimizer.step()

            # if i % n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------
            G_optimizer.zero_grad()
            # print(Gloss_pure.size() , D_model(z_nero_sampled).size() , logpz.size())
            G_ganloss = torch.log(D_model(z_nero_sampled).clamp(eps,1).squeeze())
            if args['choose'] == 0:
                Gloss = 10*Gloss_best.mean() + 10*Gloss_pure.mean() + 80*I.mean() + om.mean() - G_ganloss.mean() #+ ((0.01*Gloss_pure.detach() + I.detach() + om.detach()) * logpz.sum(1)).mean()
            elif args['choose'] == 1:
                Gloss = 10 * Gloss_best.mean() + 10 * Gloss_pure.mean() + 80 * I.mean() + om.mean() - G_ganloss.mean() + (
                            (0.01 * Gloss_pure.detach() + I.detach() + om.detach()) * logpz.sum(1)).mean()
            elif args['choose'] == 2:
                Gloss = Gloss_best.mean() + Gloss_pure.mean() + 100 * I.mean() + 100 * om.mean() - G_ganloss.mean()  # + ((0.01*Gloss_pure.detach() + I.detach() + om.detach()) * logpz.sum(1)).mean()
            elif args['choose'] == 3:
                Gloss =  Gloss_best.mean() + Gloss_pure.mean() + 100*I.mean() + 100*om.mean() - G_ganloss.mean()  + ((0.01*Gloss_pure.detach() + I.detach() + om.detach()) * logpz.sum(1)).mean()
            elif args['choose'] == 4:
                Gloss = Gloss_pure.mean()
            elif args['choose'] == 5:
                Gloss = 10 * Gloss_pure.mean() + 80 * I.mean() + om.mean() + (
                            (0.01 * Gloss_pure.detach() + I.detach() + om.detach()) * logpz.sum(1)).mean()
            # Gloss = Gloss_best.mean() + Gloss_pure.mean()+  (regu - torch.log(D_model(z_nero_sampled).squeeze()+eps) ).mean()
            Gloss.backward(retain_graph=True)
            G_optimizer.step()

            print_Gloss_total += Gloss.data
            plot_Gloss_total += Gloss.data

            print_Dloss_total += Dloss.data
            plot_Dloss_total += Dloss.data

            Glosses.append(Gloss.data)
            Dlosses.append(Dloss.data)

            if iter % print_every == 0:
                print_Gloss_avg = print_Gloss_total / print_every
                print_Gloss_total = 0
                print_Dloss_avg = print_Dloss_total / print_every
                print_Dloss_total = 0
                print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, iter / (n_iters * args['numEpochs'])),
                                                  iter, iter / n_iters * 100, print_Gloss_avg, print_Dloss_avg), end='')
                print(optional)

            if iter % plot_every == 0:
                plot_loss_avg = plot_Gloss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_Gloss_total = 0

            iter += 1
            # print(iter, datetime.datetime.now())

        res = test(textData, G_model, 'dev', max_accu)
        if res['accuracy'] > max_accu or max_accu == -1:
            print('accuracy = ', res['accuracy'], '>= min_accuracy(', max_accu, '), saving model...')
            torch.save([G_model, D_model], model_path)
            max_accu = res['accuracy']



        print('Epoch ', epoch, 'loss = ', sum(Glosses) / len(Glosses), 'Valid = ', res , 'max accuracy=',
              max_accu)

        #add data to history

        val_accuracy_history.append(res['accuracy'])
        Gloss_history.append(sum(Glosses)/len(Glosses))
        Dloss_history.append(sum(Dlosses)/len(Dlosses))
        F_macro_history.append(res['F_macro'])
        MP_history.append(res['MP'])
        MR_history.append(res['MR'])
        F_macro_n_history.append(res['F_macro_n '])
        MP_n_history.append(res['MP_n'])
        MR_n_history.append(res['MR_n'])


    #save each epochs Gloss and Dloss
    training_stats['epoch'] = list(range(len(Gloss_history)))
    training_stats['G-Loss'] = Gloss_history
    training_stats['D-Loss'] = Dloss_history
    training_stats['Val. Accuracy'] = val_accuracy_history
    training_stats['F_macro'] = F_macro_history
    training_stats['mean_precision'] = MP_history
    training_stats['mean_recall'] = MR_history
    training_stats['F_macro_negative'] = F_macro_n_history
    training_stats['MP_negative'] = MP_n_history
    training_stats['MR_negative'] = MR_n_history

    # print(training_stats)
    #save training stats to file
    pd.DataFrame(training_stats).to_csv(args['rootDir'] + '/' + args['enc_arch'] + "SMALL_training_stats_be_"+ emb_file_path+"_LM_" + args['model_arch'] + args['date'] + ".csv", index=False)

    # self.test()
    # showPlot(plot_losses)


def test(textData, model, datasetname, max_accuracy, eps = 1e-6):
    right = 0
    total = 0

    dset = []

    pppt = False
    TP_c = np.zeros(args['chargenum'])
    FP_c = np.zeros(args['chargenum'])
    FN_c = np.zeros(args['chargenum'])
    TN_c = np.zeros(args['chargenum'])
    with torch.no_grad():
        for batch in textData.getBatches(datasetname):
            x = {}
            x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs))
            x['enc_len'] = batch.encoder_lens
            x['labels'] = autograd.Variable(torch.LongTensor(batch.label)).to(args['device'])
            if args['model_arch'] in ['lstmibgan', 'lstmibgan_law']:
                x['labels'] = x['labels']

            output_probs, output_labels = model.predict(x)
            output_labels, sampled_words, wordsamplerate = output_labels

            # print("output probs shape: ", output_probs.shape)
            # print("output probs: ", output_probs)
            # print("======================================")
            # print("output labels shape: ", output_labels.shape)
            # print("output labels: ", output_labels)
            # print("===========================================")
            # print("sampled words shape: ", sampled_words.shape)
            # print("sampled words: ", sampled_words)


            if not pppt:
                pppt = True
                pind = np.random.choice(x['enc_input'].size()[0])
                for w, choice in zip(batch.encoderSeqs[pind], sampled_words[pind]):
                    # print(" ---\n")
                    # print("w is: ", w)
                    # print("choice is: ", choice)
                    if choice[1] == 1:
                        print("choice was equal to 1 \n")
                        print('<', textData.index2word[w], '>', end='')
                    else:
                        print("choice wasn't equal to 1 \n")
                        print(textData.index2word[w], end='')

                print('sample rate: ', wordsamplerate[0])
            y = F.one_hot(torch.LongTensor(x['labels'].cpu().numpy()), num_classes=args['chargenum'])  # batch c
            y = y.bool().numpy()
            answer = output_labels.cpu().numpy()
            answer = F.one_hot(torch.LongTensor(answer), num_classes=args['chargenum'])  # batch c
            answer = answer.bool().numpy()

            tp_c = ((answer == True) & (answer == y)).sum(axis=0)  # c
            fp_c = ((answer == True) & (y == False)).sum(axis=0)  # c
            fn_c = ((answer == False) & (y == True)).sum(axis=0)  # c
            tn_c = ((answer == False) & (y == False)).sum(axis=0)  # c
            TP_c += tp_c
            FP_c += fp_c
            FN_c += fn_c
            TN_c += tn_c

            batch_correct = output_labels.cpu().numpy() == x['labels'].cpu().numpy()
            # print(output_labels.size(), torch.LongTensor(batch.label).size())
            right += sum(batch_correct)
            total += x['enc_input'].size()[0]

            for ind, c in enumerate(batch_correct):
                if not c:
                    dset.append((batch.encoderSeqs[ind], x['labels'][ind], output_labels[ind]))


    res ={}

    res['accuracy'] = right / total
    P_c = TP_c / (TP_c + FP_c + eps)
    R_c = TP_c / (TP_c + FN_c + eps)
    F_c = 2 * P_c * R_c / (P_c + R_c + eps)
    res['F_macro'] = np.mean(F_c)
    res['MP'] = np.mean(P_c)
    res['MR'] = np.mean(R_c)


    P_c = TP_c / (TP_c + FP_c )
    R_c = TP_c / (TP_c + FN_c )
    F_c = 2 * P_c * R_c / (P_c + R_c )
    res['F_macro_n ']= np.nanmean(F_c)
    res['MP_n'] = np.nanmean(P_c)
    res['MR_n'] = np.nanmean(R_c)

    print("finished testing! ")

    # accuracy = res['accuracy']
    # if accuracy > max_accuracy:
    #     with open(args['rootDir'] + '/error_case_' + args['model_arch'] + args['date'] + '.txt', 'w') as wh:
    #         for d in dset:
    #             wh.write(''.join([textData.index2word[wid] for wid in d[0]]))
    #             wh.write('\t')
    #     wh.close()

    return res