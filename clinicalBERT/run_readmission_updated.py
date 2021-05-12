# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import trange, tqdm
import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

from scipy import interp



from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, confusion_matrix, \
    classification_report
# from sklearn.utils.fixes import signature
from funcsigs import signature
import matplotlib.pyplot as plt
import seaborn as sns

import time
import datetime

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from pytorch_pretrained_bert.optimization import BertAdam
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_csv(cls, input_file):
        """Reads a comma separated value file."""
        file = pd.read_csv(input_file)
        lines = zip(file.ID, file.TEXT, file.Label)
        return lines





class readmissionProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.csv")))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "val.csv")), "val")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = str(int(line[2]))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

def preprocess_tokenize_embed_data(df_filename, max_seq_length):
    '''
    Function to use encode_plus to process text data and lables
            # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.

    input = data file name containing text data/sentences under TEXT column and corresponding labels for classification task

    output = input_ids (encoded sentence id), attention_mask (mask ids for BERT), labels - all converted to tensors
    '''
    tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    df = pd.read_csv(df_filename)

    print('Number of sentences: {:,}\n'.format(df.shape[0]))

    sentences = df.TEXT.values
    labels = df.Label.values

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in tqdm(sentences):
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_seq_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    # labels = torch.tensor(labels)
    labels = torch.tensor(labels, dtype=torch.long)
    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])

    # will teturn a TensorDataset using these input_ids, attention_masks and labels
    print("returning TensorDataset! ")
    return TensorDataset(input_ids, attention_masks, labels)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def vote_score(df, score, args):
    df['pred_score'] = score
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
    plt.show()
    string = f'auroc_clinicalbert_{args.readmission_mode}.png'
    plt.savefig(os.path.join(args.output_dir, string))

    return fpr, tpr, df_out


def pr_curve_plot(y, y_score, args):
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

    string = f'auprc_clinicalbert_{args.readmission_mode}.png'

    plt.savefig(os.path.join(args.output_dir, string))


def vote_pr_curve(df, score, args):
    df['pred_score'] = score
    df_sort = df.sort_values(by=['ID'])
    # score
    temp = (df_sort.groupby(['ID'])['pred_score'].agg(max) + df_sort.groupby(['ID'])['pred_score'].agg(sum) / 2) / (
                1 + df_sort.groupby(['ID'])['pred_score'].agg(len) / 2)
    y = df_sort.groupby(['ID'])['Label'].agg(np.min).values

    precision, recall, thres = precision_recall_curve(y, temp)
    pr_thres = pd.DataFrame(data=list(zip(precision, recall, thres)), columns=['prec', 'recall', 'thres'])
    vote_df = pd.DataFrame(data=list(zip(temp, y)), columns=['score', 'label'])

    pr_curve_plot(y, temp, args)

    temp = pr_thres[pr_thres.prec > 0.799999].reset_index()

    rp80 = 0
    if temp.size == 0:
        print('Test Sample too small or RP80=0')
    else:
        rp80 = temp.iloc[0].recall
        print('Recall at Precision of 80 is {}', rp80)

    return rp80


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--readmission_mode", default=None, type=str, help="early notes or discharge summary")

    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=2,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument("--date",
                        default=None,
                        type=str,
                        required=True,
                        help="The date in whichever format to be appended to filenames.")

    args = parser.parse_args()

    processors = {
        "readmission": readmissionProcessor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    # tokenizer = BertTokenizer.from_pretrained(args.bert_model)


    train_examples = None
    num_train_steps = None
    if args.do_train:
        print("rabbleeeeeeee")
        # train_examples = processor.get_train_examples(args.data_dir)

        train_dataset = preprocess_tokenize_embed_data(f"{args.data_dir}train.csv", args.max_seq_length)
        val_dataset = preprocess_tokenize_embed_data(f"{args.data_dir}val.csv", args.max_seq_length)
        num_train_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        # val_examples = processor.get_dev_examples((args.data_dir))
        
    if args.bert_model == "emilyalsentzer/Bio_ClinicalBERT":
        print("getting pretrained emily")
        # Prepare model
        model = BertForSequenceClassification.from_pretrained(

            "emilyalsentzer/Bio_ClinicalBERT",  # Use the pretrained BioClinicalBERT from: "emilyalsentzer/Bio_ClinicalBERT"
            num_labels=1,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
    else:
        model = BertForSequenceClassification.from_pretrained(

            args.bert_model,
            # Use the pretrained BioClinicalBERT from: "emilyalsentzer/Bio_ClinicalBERT"
            num_labels=1,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )

    print(model)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # if args.do_train:
    #     # I believe the 'W' stands for 'Weight Decay fix"
    #     #optimizer = AdamW(model.parameters(),
    #                       # lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
    #                       # eps=1e-8  # args.adam_epsilon  - default is 1e-8.
    #                       # )
    #
    #     # Prepare optimizer
    #     # if args.fp16:
    #     #     param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
    #     #                        for n, param in model.named_parameters()]
    #     # elif args.optimize_on_cpu:
    #     #     param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
    #     #                        for n, param in model.named_parameters()]
    #     # else:
    #     #     param_optimizer = list(model.named_parameters())



    global_step = 0
    train_loss = 100000
    number_training_steps = 1
    global_step_check = 0
    train_loss_history = []
    training_stats = []
    average_epo_loss_history = []
    if args.do_train:

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
        # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order.
        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler=train_sampler,  # Select batches randomly
            batch_size=args.train_batch_size  # Trains with this batch size.
        )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
            val_dataset,  # The validation samples.
            sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
            batch_size=args.train_batch_size  # Evaluate with this batch size.
        )
        # set up scheduler
        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * args.num_train_epochs


        num_train_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        # Create the learning rate scheduler.
        # scheduler = get_linear_schedule_with_warmup(optimizer,
        #                                             num_warmup_steps=0,  # Default value in run_glue.py
        #                                             num_training_steps=total_steps)

        # Prepare optimizer
        if args.fp16:
            param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                                for n, param in model.named_parameters()]
        elif args.optimize_on_cpu:
            param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                                for n, param in model.named_parameters()]
        else:
            param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps)
        for epoch_i in trange(int(args.num_train_epochs), desc="Epoch"):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1,args.num_train_epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 500 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                #         print(b_input_ids.dtype)
                #         print(b_labels.dtype)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # In PyTorch, calling `model` will in turn call the model's `forward`
                # function and pass down the arguments. The `forward` function is
                # documented here:
                # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
                # The results are returned in a results object, documented here:
                # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
                # Specifically, we'll get the loss (because we provided labels) and the
                # "logits"--the model outputs prior to activation.
                result = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               return_dict=True)

                loss = result.loss
                logits = result.logits

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                train_loss_history.append(loss.item())
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1

                ##=================================================================##
                ##==== CODE BELOW IS FROM NEWER VERSION FOR A 2 CLASS PROBLEM =====##

                # #         loss, logits = model(b_input_ids,
                # #                          token_type_ids=None,
                # #                          attention_mask=b_input_mask,
                # #                          labels=b_labels)
                #
                # #         print(loss)
                # #         print(logits)
                # # Accumulate the training loss over all of the batches so that we can
                # # calculate the average loss at the end. `loss` is a Tensor containing a
                # # single value; the `.item()` function just returns the Python value
                # # from the tensor.
                # total_train_loss += loss.item()
                #
                # # Perform a backward pass to calculate the gradients.
                # loss.backward()
                #
                # # Clip the norm of the gradients to 1.0.
                # # This is to help prevent the "exploding gradients" problem.
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                #
                # # Update parameters and take a step using the computed gradient.
                # # The optimizer dictates the "update rule"--how the parameters are
                # # modified based on their gradients, the learning rate, etc.
                # optimizer.step()
                #
                # # Update the learning rate.
                # # scheduler.step()
                ##==========================================================================###
            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()
            m = nn.Sigmoid()
            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            val_loss, val_accuracy = 0, 0
            nb_val_steps, nb_val_examples = 0, 0

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in tqdm(validation_dataloader):
                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                #         print(b_input_ids.dtype)
                #         print(b_labels.dtype)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    result = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels,
                                   return_dict=True)
                #             loss, logits = model(b_input_ids,
                #                      token_type_ids=None,
                #                      attention_mask=b_input_mask,
                #                      labels=b_labels)

                # Get the loss and "logits" output by the model. The "logits" are the
                # output values prior to applying an activation function like the
                # softmax.
                tmp_val_loss = result.loss
                logits = result.logits

                logits = torch.squeeze(m(logits)).detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()

                outputs = np.asarray([1 if i else 0 for i in (logits.flatten() >= 0.5)])
                tmp_val_accuracy = np.sum(outputs == label_ids)

                true_labels = true_labels + label_ids.flatten().tolist()
                pred_labels = pred_labels + outputs.flatten().tolist()

                val_loss += tmp_val_loss.mean().item()
                val_accuracy += tmp_val_accuracy

                nb_val_examples += b_input_ids.size(0)
                nb_val_steps += 1

                # Accumulate the validation loss.
                total_eval_loss += tmp_val_loss.item()

                # # Move logits and labels to CPU
                # logits = logits.detach().cpu().numpy()
                # label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                # total_eval_accuracy += flat_accuracy(logits, label_ids)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("Training complete!")


        string = f'{args.output_dir}/pytorch_model_new_{args.readmission_mode}_{args.date}.bin'
        torch.save(model.state_dict(), string)

        training_stats_df = pd.DataFrame(training_stats)
        training_stats_df.to_csv(f'{args.output_dir}/training_val_metrics.csv', index=False)

        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)

        # Plot the learning curve.
        plt.plot(training_stats_df['Training Loss'], 'b-o', label="Training")
        plt.plot(training_stats_df['Valid. Loss'], 'g-o', label="Validation")


        # Label the plot.
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        # plt.xticks([1, 2, 3, 4])

        plt.show()
        plt.savefig(f'{args.output_dir}train_val_loss_figure.png')

        fig3 = plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        x_range = np.arange(epoch_i+1)
        plt.plot(x_range, training_stats_df['Training Loss'], label='train loss')
        plt.plot(x_range, training_stats_df['Valid. Loss'], label='validation loss')
        plt.title('Losses')
        plt.legend()

        plt.subplot(1, 2, 2)
        x_range = np.arange(epoch_i+1)
        plt.plot(x_range, training_stats_df['Valid. Accur.'], label='val acc')
        plt.title('Accuracy')
        plt.legend()
        fig3.savefig(f'{args.output_dir}/train_val_loss_val_acc_{args.readmission_mode}_{args.date}.png', dpi=fig3.dpi)

    m = nn.Sigmoid()
    if args.do_eval:
        test_dataset = preprocess_tokenize_embed_data(f"{args.data_dir}test.csv", args.max_seq_length)

        # Set the batch size.
        batch_size = 32

        # Create the DataLoader.

        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)
        # Put model in evaluation mode
        model.eval()

        # Tracking variables
        logits_history, pred_labels, true_labels = [], [], []
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in tqdm(test_dataloader):
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            #         print(b_input_ids.dtype)
            #         print(b_labels.dtype)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                result = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               return_dict=True)
            #             loss, logits = model(b_input_ids,
            #                      token_type_ids=None,
            #                      attention_mask=b_input_mask,
            #                      labels=b_labels)

            # Get the loss and "logits" output by the model. The "logits" are the
            # output values prior to applying an activation function like the
            # softmax.
            tmp_eval_loss = result.loss
            logits = result.logits

            # Accumulate the evaluation loss.
            total_eval_loss += tmp_eval_loss.item()
            logits = torch.squeeze(m(logits)).detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            print("logits shape: ", logits.shape)
            print("label shape: ", label_ids.shape)

            outputs = np.asarray([1 if i else 0 for i in (logits.flatten() >= 0.5)])
            tmp_eval_accuracy = np.sum(outputs == label_ids)

            true_labels = true_labels + label_ids.flatten().tolist()
            pred_labels = pred_labels + outputs.flatten().tolist()
            logits_history = logits_history + logits.flatten().tolist()

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        df = pd.DataFrame({'logits': logits_history, 'pred_label': pred_labels, 'label': true_labels})

        print("logit history shape: ", df.logits.shape)
        string = 'logits_clinicalbert_' + args.readmission_mode + '_chunks.csv'
        df.to_csv(os.path.join(args.output_dir, string))

        df_test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
        print(df_test.shape)

        fpr, tpr, df_out = vote_score(df_test, logits_history, args)

        string = 'logits_clinicalbert_' + args.readmission_mode + '_readmissions.csv'
        df_out.to_csv(os.path.join(args.output_dir, string))

        rp80 = vote_pr_curve(df_test, logits_history, args)

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step_check,
                  'training loss': train_loss / number_training_steps,
                  'RP80': rp80}

        # result_df = pd.DataFrame(result)
        # result_df.to_csv(f'{args.output_dir}')
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        #

        # preds = np.argmax(preds, axis = 1)
        print(classification_report(true_labels, pred_labels))

        cf = confusion_matrix(true_labels, pred_labels, normalize='true')
        df_cf = pd.DataFrame(cf, ['not r/a', 'readmitted'], ['not r/a', 'readmitted'])
        plt.figure(figsize=(6, 6))
        plt.suptitle("Readmitted vs not readmitted")
        sns.heatmap(df_cf, annot=True, cmap='Blues')
        plt.savefig(f"{args.output_dir}Confusion_Matrix.png")




        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
