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


class Batch:
    """
    Struct containing batches info
    """

    def __init__(self):
        self.encoderSeqs = []
        self.encoder_lens = []
        self.label = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.decoder_lens = []
        self.rationals = []
        self.raw = []


class TextDataMimic:
    """Dataset class
    Warning: No vocabulary limit
    """

    def __init__(self, corpusname, datadir, taskname, trainLM=False, test_phase=True, big_emb=False, new_emb = False):

        """Load all conversations
        Args:
            args: parameters of the model
        """

        # Path variables
        if corpusname == 'cail':
            self.tokenizer = lambda x: list(jieba.cut(x))
        elif corpusname == 'mimic':
            # self.tokenizer = word_tokenize
            self.tokenizer = nltk.RegexpTokenizer(r"\w+").tokenize


        self.datadir = datadir
        self.taskname = taskname
        self.basedir = './data/mimic3/'

        #use if on local pc
        # self.embfile = "../clinicalBERT/word2vec+fastText/word2vec+fastText/word2vec.model"
        # self.big_embfile = "../clinicalBERT/word2vec+fastText/word2vec+fastText/BioWordVec_PubMed_MIMICIII_d200.vec.bin"
        #use if on vm
        # self.embfile = "../clinicalBERT/word2vec+fastText/word2vec+fastText/word2vec.model"
        self.big_emb = big_emb
        self.new_emb = new_emb
        print("self.new emb in textData is: ", self.new_emb)
        print("self.big emb in textData is: ", self.big_emb)
        if self.new_emb:
            print("using new strict embeddings")
            self.embfile = "./data/mimic3/new_mimic_word2vec_200_strict.model"

        else:
            print("using original embeddings")
            self.embfile = "../clinicalBERT/word2vec+fastText/word2vec+fastText/word2vec.model"
        # self.embfile = embedding_file
        print(f"using this embedding model:{self.embfile} ")
        # self.embfile = "./data/mimic3/new_mimic_word2vec.model"
        self.big_embfile = "../clinicalBERT/word2vec+fastText/BioWordVec_PubMed_MIMICIII_d200.vec.bin"


        if test_phase:
            self.test_phase = True
        else:
            self.test_phase = False

        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]
        if not trainLM:
            self.datasets = self.loadCorpus_Mimic3()
        else:
            # TODO need to adapt below to load mimic3 data in a way ready for the language modelling (if we end up using)
            self.datasets = self.load_all_mimic()

        print('set')
        # Plot some stats:
        self._printStats(corpusname)

        #         if args['playDataset']:
        #             self.playDataset()

        self.batches = {}

    def loadCorpus_Mimic3(self):
        """
        Load/create the mimic 3 dataset
        """

        #         self.datadir = '../clinicalBERT/data/discharge/'
        self.corpus_file_train = self.datadir + self.taskname + '/train.csv'
        self.corpus_file_dev = self.datadir + self.taskname + '/val.csv'
        self.corpus_file_test = self.datadir + self.taskname + '/test.csv'

        if self.big_emb:
            self.data_dump_path = f"{self.basedir}/mimic3_processed_bigembed_{self.taskname}.pkl"
        elif self.new_emb:
            self.data_dump_path = f"{self.basedir}/mimic3_processed_new200strict_{self.taskname}.pkl"
        else:
            self.data_dump_path = f"{self.basedir}/mimic3_processed_originalembs_{self.taskname}.pkl"

        print(self.data_dump_path)
        datasetExist = os.path.isfile(self.data_dump_path)

        if not datasetExist:  # First time we load the database: creating all files
            print('Training data not found. Creating dataset...')

            total_words = []

            # index2word, index2vec, word2vec, key2index
            if self.big_emb:
                print("using big boy embeddings!")
                self.org_index2word, self.org_index2vec, self.org_word2vec, self.org_key2index = self.get_word2vec_from_pretrained(
                    self.big_embfile)
            if self.new_emb:
                print("using new 200d embeddings from: ", self.embfile)
                self.org_index2word, self.org_index2vec, self.org_word2vec, self.org_key2index = self.get_word2vec_from_pretrained(
                    self.embfile)

            else:
                print("using original embeddings! from: ", self.embfile)
                self.org_index2word, self.org_index2vec, self.org_word2vec, self.org_key2index = self.get_word2vec_from_pretrained(
                    self.embfile)

            # need to re order these to have special tokens same as beer dataset
            # ord_word2index, ord_index2word, ord_index2vector
            self.word2index, self.index2word, self.index2vector = self.rearrange_word2vec(self.org_index2word,
                                                                                          self.org_index2vec,
                                                                                          self.org_word2vec,
                                                                                          self.org_key2index)
            # get the set of these index2words - words in index position essentially
            self.index2word_set = set(self.index2word)

            #             print("self index 2 word :", self.index2word)
            #             print("self word 2 index : ", self.word2index)

            datasets = self.format_mimic_datasets()

            print(len(datasets['train']), len(datasets['dev']), len(datasets['test']))

            #             # self.raw_sentences = copy.deepcopy(dataset)
            #             for setname in ['train', 'dev', 'test']:
            #                 dataset[setname] = [(self.TurnWordID(sen), y, sen, rational) for sen, y, rational in tqdm(dataset[setname])]

            # Saving
            print('Saving dataset...')
            self.save_mimic_datasets(datasets, self.data_dump_path)  # Saving tf samples
        else:
            print(f"Found already saved data at {self.data_dump_path}! Loading that instead")
            datasets = self.loadDataset(self.data_dump_path)
            print('loaded')

        return datasets

    def loadDataset(self, filename):
        """
        Load samples from file
        Args:
            filename (str): pickle filename

        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        if self.big_emb:
            with open(dataset_path, 'rb') as handle:
                data = joblib.load(handle)  # Warning: If adding something here, also modifying saveDataset
                self.word2index = data['word2index']
                self.index2word = data['index2word']
                self.index2vector = data['index2vector']
                datasets = data['datasets']
            if self.test_phase:
                test_datasets = {}
                test_datasets['train'] = datasets['train'][0:2000]
                test_datasets['dev'] = datasets['dev'][0:1000]
                test_datasets['test'] = datasets['test'][0:1000]
                self.index2word_set = set(self.index2word)
                print('training: \t', len(test_datasets['train']))
                print('dev: \t', len(test_datasets['dev']))
                print('testing: \t', len(test_datasets['test']))
                self.index2word_set = set(self.index2word)
                print('w2i shape: ', len(self.word2index))
                print('i2w shape: ', len(self.index2word))
                print('embeding shape: ', self.index2vector.shape)

                return test_datasets
            else:
                print('training: \t', len(datasets['train']))
                print('dev: \t', len(datasets['dev']))
                print('testing: \t', len(datasets['test']))
                self.index2word_set = set(self.index2word)
                print('w2i shape: ', len(self.word2index))
                print('i2w shape: ', len(self.index2word))
                print('embeding shape: ', self.index2vector.shape)
                return datasets
        else:
            with open(dataset_path, 'rb') as handle:
                data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
                self.word2index = data['word2index']
                self.index2word = data['index2word']
                self.index2vector = data['index2vector']
                datasets = data['datasets']
            if self.test_phase:
                test_datasets = {}
                test_datasets['train'] = datasets['train'][0:500]
                test_datasets['dev'] = datasets['dev'][0:500]
                test_datasets['test'] = datasets['test'][0:500]
                self.index2word_set = set(self.index2word)
                print('training: \t', len(test_datasets['train']))
                print('dev: \t', len(test_datasets['dev']))
                print('testing: \t', len(test_datasets['test']))
                self.index2word_set = set(self.index2word)
                print('w2i shape: ', len(self.word2index))
                print('i2w shape: ', len(self.index2word))
                print('embeding shape: ', self.index2vector.shape)

                return test_datasets
            else:
                print('training: \t', len(datasets['train']))
                print('dev: \t', len(datasets['dev']))
                print('testing: \t', len(datasets['test']))
                self.index2word_set = set(self.index2word)
                print('w2i shape: ', len(self.word2index))
                print('i2w shape: ', len(self.index2word))
                print('embeding shape: ', self.index2vector.shape)
                return datasets

    def get_word2vec_from_pretrained(self, filename):

        if self.big_emb:
            model = gensim.models.KeyedVectors.load_word2vec_format(f'{filename}', binary=True)
            # load in gensim format
            weights = model

            # convert to tensor for pytorch
            weights = torch.FloatTensor(model.vectors)
            print("weights vector shape: ", weights.shape)

            #get subset of the embeddings - just 100,000 for now

            # convert to embedding layer
            embedding = nn.Embedding.from_pretrained(weights[0:100000])

            print("embedding tensor shape: ", embedding)

            words = model.index_to_key
            # get the word2vec dictionary {'word':vector}
            word2vec = {word: model[word] for word in words}

            # get the vec2index array - essentially the vector arrays are in the index position corresponding to the word in word2vec
            index2vec = model.vectors

            # index to words - just a list of words in correct index position

            index2word = words
            print("length of indices to words: ", len(index2word))

            key2index = model.key_to_index
            #         self.index2word = index2word
            #         self.index2vec = index2vec
            #         self.word2vec = word2vec
            return index2word, index2vec, word2vec, key2index
        else:
            model = gensim.models.KeyedVectors.load(f'{filename}')

            # load in gensim format
            weights = model.wv

            # convert to tensor for pytorch
            weights = torch.FloatTensor(model.wv.vectors)
            print("weights vector shape: ", weights.shape)

            # convert to embedding layer
            embedding = nn.Embedding.from_pretrained(weights)

            print("embedding tensor shape: ", embedding)

            words = model.wv.index_to_key
            # get the word2vec dictionary {'word':vector}
            word2vec = {word: model.wv[word] for word in words}

            # get the vec2index array - essentially the vector arrays are in the index position corresponding to the word in word2vec
            index2vec = model.wv[model.wv.index_to_key]

            # index to words - just a list of words in correct index position

            index2word = words
            print("length of indices to words: ", len(index2word))

            key2index = model.wv.key_to_index
            #         self.index2word = index2word
            #         self.index2vec = index2vec
            #         self.word2vec = word2vec
            return index2word, index2vec, word2vec, key2index

    def rearrange_word2vec(self, index2word, index2vec, word2vec, key2index):

        word2index = dict()
        # if using the bigger embeddings - we already have tokens for fullstops - so only want to specify these special tokens
        if self.big_emb:
            print("re-arranging the big embeddings")
            word2index['PAD'] = 0
            word2index['START_TOKEN'] = 1
            word2index['END_TOKEN'] = 2
            word2index['UNK'] = 3

            # start the counter/embedding ID at 1 more than the newly added special tokens
            cnt = 4
            index2vector = []
            for word in word2vec:
                index2vector.append(word2vec[word])

                word2index[word] = cnt
                #             print(word, cnt)
                cnt += 1
            vectordim = len(word2vec[word])
            print('before add special token:', len(index2vector))
            index2vector = [np.random.normal(size=[vectordim]).astype('float32') for _ in range(4)] + index2vector
            print('after add special token:', len(index2vector))
            index2vector = np.asarray(index2vector)
            index2word = [w for w, n in word2index.items()]
            print(len(word2index), cnt)
            print('Dictionary Got!')
            return word2index, index2word, index2vector
        elif self.new_emb:
            print("re-arranging the newer embeddings")
            # for the smaller embeddings file - it did not seem to know fullstops - so we explicitly add it here
            word2index['PAD'] = 0
            word2index['START_TOKEN'] = 1
            word2index['END_TOKEN'] = 2
            word2index['UNK'] = 3
            #             word2index['.'] = 4
            # word2index['PAD'] = 1
            # word2index['UNK'] = 0

            cnt = 4
            index2vector = []
            for word in word2vec:
                index2vector.append(word2vec[word])

                word2index[word] = cnt
                #             print(word, cnt)
                cnt += 1
            vectordim = len(word2vec[word])
            print('before add special token:', len(index2vector))
            index2vector = [np.random.normal(size=[vectordim]).astype('float32') for _ in range(4)] + index2vector
            print('after add special token:', len(index2vector))
            index2vector = np.asarray(index2vector)
            index2word = [w for w, n in word2index.items()]
            print(len(word2index), cnt)
            print('Dictionary Got!')
            return word2index, index2word, index2vector
        else:
            print("re-arranging the smaller embeddings")
            # for the smaller embeddings file - it did not seem to know fullstops - so we explicitly add it here
            word2index['PAD'] = 0
            word2index['START_TOKEN'] = 1
            word2index['END_TOKEN'] = 2
            word2index['UNK'] = 3
            word2index['.'] = 4
            # word2index['PAD'] = 1
            # word2index['UNK'] = 0

            cnt = 5
            index2vector = []
            for word in word2vec:
                index2vector.append(word2vec[word])

                word2index[word] = cnt
                #             print(word, cnt)
                cnt += 1
            vectordim = len(word2vec[word])
            print('before add special token:', len(index2vector))
            index2vector = [np.random.normal(size=[vectordim]).astype('float32') for _ in range(5)] + index2vector
            print('after add special token:', len(index2vector))
            index2vector = np.asarray(index2vector)
            index2word = [w for w, n in word2index.items()]
            print(len(word2index), cnt)
            print('Dictionary Got!')
            return word2index, index2word, index2vector

    # ord_word2index, ord_index2word, ord_index2vector = rearrange_word2vec(index2word, index2vec, word2vec, key2index)

    def get_word_ids(self, text):

        #     print("getting word ids for each token in provided sentences!")

        #         index2word_set = set(self.index2word)
        #     print(text)
        res = []
        for token in text:
            #         print(token)
            if token in self.index2word_set:
                #             print(token)
                #             print(ord_word2index[token])
                word_id = self.word2index[token]
                res.append(word_id)
            else:
                res.append(self.word2index["UNK"])

        return res

    # test_data["word_ids"] = test_data['tokenized_text'].apply(get_word_ids)

    def format_mimic_datasets(self):

        dataset = dict()

        dataset["train"] = pd.read_csv(self.corpus_file_train, index_col=0)
        #     train_df["word_ids"] =

        dataset["dev"] = pd.read_csv(self.corpus_file_dev, index_col=0)

        dataset["test"] = pd.read_csv(self.corpus_file_test, index_col=0)

        setnames = ["train", "dev", "test"]

        for setname in setnames:
            print("working on: ", setname)
            df = dataset[setname]
            print(df.head())
            df["tokenized_text"] = df.TEXT.apply(self.tokenizer)
            df["word_ids"] = df["tokenized_text"].apply(self.get_word_ids)
            df["rational"] = -1
            dataset[setname] = np.asarray(df[["word_ids", 'Label', 'tokenized_text', "rational"]])

        print("training data size: ", dataset["train"].shape)
        print("dev data size: ", dataset["dev"].shape)
        print("test data size: ", dataset["test"].shape)
        return dataset

    def save_mimic_datasets(self, datasets, dump_path):

        all_data = {}
        all_data["word2index"] = self.word2index
        all_data["index2word"] = self.index2word
        all_data["index2vector"] = self.index2vector

        all_data["datasets"] = datasets

        #         # Create output directory if needed
        #         if not os.path.exists(self.):
        #             os.makedirs(output_dir)

        if self.big_emb:
            with open(f'{dump_path}', 'wb') as handle:
                joblib.dump(all_data, handle)
        else:
            with open(f'{dump_path}', 'wb') as handle:
                pickle.dump(all_data, handle, -1)

    def load_all_mimic(self):
        if self.new_emb:
            self.data_dump_path1 = self.basedir + '/mimic3_processed_new200strict_3days.pkl'
            self.data_dump_path2 = self.basedir + '/mimic3_processed_new200strict_discharge.pkl'
            self.data_dump_all_path = self.basedir + '/mimic3_processed_new200strict_all.pkl'

        else:
            self.data_dump_path1 = self.basedir + '/mimic3_processed_originalembs_3days.pkl'
            self.data_dump_path2 = self.basedir + '/mimic3_processed_originalembs_discharge.pkl'

            self.data_dump_all_path = self.basedir + '/mimic3_processed_originalembs_all.pkl'

        datasetExist = os.path.isfile(self.data_dump_all_path)

        if not datasetExist:  # First time we load the database: creating all files
            print('Already processed combined data not found. Creating dataset of all combined...')

            data = {'train': [], 'dev': [], 'test': []}
            d1 = self.loadDataset(self.data_dump_path1)
            d2 = self.loadDataset(self.data_dump_path2)

            # index2word, index2vec, word2vec, key2index
            self.org_index2word, self.org_index2vec, self.org_word2vec, self.org_key2index = self.get_word2vec_from_pretrained(
                self.embfile)

            # need to re order these to have special tokens same as beer dataset
            # ord_word2index, ord_index2word, ord_index2vector
            self.word2index, self.index2word, self.index2vector = self.rearrange_word2vec(self.org_index2word,
                                                                                          self.org_index2vec,
                                                                                          self.org_word2vec,
                                                                                          self.org_key2index)
            # get the set of these index2words - words in index position essentially
            self.index2word_set = set(self.index2word)

            data['train'] = np.asarray(pd.concat([pd.DataFrame(d1['train']), pd.DataFrame(d2['train'])]))
            data['dev'] = np.asarray(pd.concat([pd.DataFrame(d1['dev']), pd.DataFrame(d2['dev'])]))
            data['test'] = np.asarray(pd.concat([pd.DataFrame(d1['test']), pd.DataFrame(d2['test'])]))

            # save
            print("saving the combined mimic dataset")
            self.save_mimic_datasets(data, self.data_dump_all_path)
        else:
            print("already combined them so loading that instead! ")
            data = self.loadDataset(self.data_dump_all_path)

        return data

    def _printStats(self, corpusname):
        print('Loaded {}: {} words, {} QA'.format(corpusname, len(self.word2index), len(self.trainingSamples)))

    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.datasets['train'])

    def getSampleSize(self, setname='train'):
        """Return the size of the dataset
        Return:
            int: Number of training samples
        """
        return len(self.datasets[setname])

    def getVocabularySize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2index)

    def sequence2str(self, sequence, clean=False, reverse=False):
        """Convert a list of integer into a human readable string
        Args:
            sequence (list<int>): the sentence to print
            clean (Bool): if set, remove the <go>, <pad> and <eos> tokens
            reverse (Bool): for the input, option to restore the standard order
        Return:
            str: the sentence
        """

        if not sequence:
            return ''

        if not clean:
            return ' '.join([self.index2word[idx] for idx in sequence])

        sentence = []
        for wordId in sequence:
            if wordId == self.word2index['END_TOKEN']:  # End of generated sentence
                break
            elif wordId != self.word2index['PAD'] and wordId != self.word2index['START_TOKEN']:
                sentence.append(self.index2word[wordId])

        if reverse:  # Reverse means input so no <eos> (otherwise pb with previous early stop)
            sentence.reverse()

        return self.detokenize(sentence)

    def detokenize(self, tokens):
        """Slightly cleaner version of joining with spaces.
        Args:
            tokens (list<string>): the sentence to print
        Return:
            str: the sentence
        """
        return ''.join([
            ' ' + t if not t.startswith('\'') and
                       t not in string.punctuation
            else t
            for t in tokens]).strip().capitalize()

    def batchSeq2str(self, batchSeq, seqId=0, **kwargs):
        """Convert a list of integer into a human readable string.
        The difference between the previous function is that on a batch object, the values have been reorganized as
        batch instead of sentence.
        Args:
            batchSeq (list<list<int>>): the sentence(s) to print
            seqId (int): the position of the sequence inside the batch
            kwargs: the formatting options( See sequence2str() )
        Return:
            str: the sentence
        """
        sequence = []
        for i in range(len(batchSeq)):  # Sequence length
            sequence.append(batchSeq[i][seqId])
        return self.sequence2str(sequence, **kwargs)

    def sentence2batch(self, sentence):
        """Encode a sequence and return a batch as an input for the model
        Return:
            Batch: a batch object containing the sentence, or none if something went wrong
        """
        print("the sentence was: ", sentence)
        if sentence == '':
            return None

        # First step: Divide the sentence in token
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) > 512:
            return None

        print("tokens are: ", tokens)
        # Second step: Convert the token in word ids
        #         wordIds = []
        #         for token in tokens:
        #             print(f"converting token {token} to word ID")
        #             wordIds.append(self.get_word_ids(token))  # Create the vocabulary and the training sentences

        wordIds = self.get_word_ids(tokens)

        print("wordIds after gettitng ids: ", wordIds)
        # Third step: creating the batch (add padding, reverse)
        #         batch = self._createBatch([[wordIds, []]])  # Mono batch, no target output
        batch = Batch()

        batch.encoderSeqs.append(wordIds)
        batch.encoder_lens.append(len(wordIds))
        batch.label.append([0])
        batch.rationals.append([-1])
        batch.raw.append(tokens)

        maxlen_enc = 512
        batch.encoderSeqs[0] = batch.encoderSeqs[0] + [self.word2index['PAD']] * (
                maxlen_enc - len(batch.encoderSeqs[0]))

        return batch

    def deco2sentence(self, decoderOutputs):
        """Decode the output of the decoder and return a human friendly sentence
        decoderOutputs (list<np.array>):
        """
        sequence = []

        # Choose the words with the highest prediction score
        for out in decoderOutputs:
            sequence.append(np.argmax(out))  # Adding each predicted word ids

        return sequence  # We return the raw sentence. Let the caller do some cleaning eventually

    def _createBatch(self, samples):
        """Create a single batch from the list of sample. The batch size is automatically defined by the number of
        samples given.
        The inputs should already be inverted. The target should already have <go> and <eos>
        Warning: This function should not make direct calls to args['batchSize'] !!!
        Args:
            samples (list<Obj>): a list of samples, each sample being on the form [input, target]
        Return:
            Batch: a batch object en
        """
        # print("creating batch!")
        batch = Batch()
        batchSize = len(samples)
        # print("batchsize is: ", batchSize)

        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            sen_ids, y, raw_sen, rational = samples[i]

            if len(sen_ids) > 512:
                sen_ids = sen_ids[:512]

            batch.encoderSeqs.append(sen_ids)
            batch.encoder_lens.append(len(batch.encoderSeqs[i]))
            batch.label.append(y)
            batch.rationals.append(rational)
            batch.raw.append(raw_sen)
            # print(y)

        maxlen_enc = max(batch.encoder_lens)

        for i in range(batchSize):
            batch.encoderSeqs[i] = batch.encoderSeqs[i] + [self.word2index['PAD']] * (
                    maxlen_enc - len(batch.encoderSeqs[i]))

        return batch

    def getBatches(self, setname='train'):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        print("getting batches for :", setname)
        print("batch size is: ", args['batchSize'])
        if setname not in self.batches:
            # self.shuffle()

            batches = []
            print(setname, 'size:', len(self.datasets[setname]))

            def genNextSamples():
                """ Generator over the mini-batch training samples
                """
                for i in range(0, self.getSampleSize(setname), args['batchSize']):
                    yield self.datasets[setname][i:min(i + args['batchSize'], self.getSampleSize(setname))]



            # TODO: Should replace that by generator (better: by tf.queue)

            for index, samples in enumerate(genNextSamples()):
                # print([self.index2word[id] for id in samples[5][0]], samples[5][2])

                batch = self._createBatch(samples)
                batches.append(batch)

            self.batches[setname] = batches

        # print([self.index2word[id] for id in batches[2].encoderSeqs[5]], batches[2].raws[5])
        return self.batches[setname]

    def _createBatch_forLM(self, samples):
        """Create a single batch from the list of sample. The batch size is automatically defined by the number of
        samples given.
        The inputs should already be inverted. The target should already have <go> and <eos>
        Warning: This function should not make direct calls to args['batchSize'] !!!
        Args:
            samples (list<Obj>): a list of samples, each sample being on the form [input, target]
        Return:
            Batch: a batch object en
        """

        batch = Batch()
        batchSize = len(samples)

        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            sen_ids = samples[i]
            if len(sen_ids) > args['maxLengthEnco']:
                sen_ids = sen_ids[:args['maxLengthEnco']]
            batch.decoderSeqs.append([self.word2index['START_TOKEN']] + sen_ids)
            batch.decoder_lens.append(len(batch.decoderSeqs[i]))
            batch.targetSeqs.append(sen_ids + [self.word2index['END_TOKEN']])

        # print(batch.decoderSeqs)
        # print(batch.decoder_lens)
        maxlen_dec = max(batch.decoder_lens)
        maxlen_dec = min(maxlen_dec, args['maxLengthEnco'])

        for i in range(batchSize):
            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.word2index['PAD']] * (
                        maxlen_dec - len(batch.decoderSeqs[i]))
            batch.targetSeqs[i] = batch.targetSeqs[i] + [self.word2index['PAD']] * (
                        maxlen_dec - len(batch.targetSeqs[i]))

        return batch

    def getBatches_forLM(self, setname='train'):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        if setname not in self.batches:
            self.shuffle()

            dataset_sen = self.paragraph2sentence(self.datasets[setname])
            sennum = len(dataset_sen)
            print(sennum)

            batches = []
            print(len(self.datasets[setname]))

            def genNextSamples():
                """ Generator over the mini-batch training samples
                """
                for i in range(0, sennum, args['batchSize']):
                    yield dataset_sen[i:min(i + args['batchSize'], sennum)]

            # TODO: Should replace that by generator (better: by tf.queue)

            for index, samples in enumerate(genNextSamples()):
                # print([self.index2word[id] for id in samples[5][0]], samples[5][2])
                batch = self._createBatch_forLM(samples)
                batches.append(batch)

            self.batches[setname] = batches

        # print([self.index2word[id] for id in batches[2].encoderSeqs[5]], batches[2].raws[5])
        return self.batches[setname]

    def paragraph2sentence(self, doclist):
        split_tokens = [self.word2index['.']]
        print("split tokens: ", split_tokens)
        sen_list = []
        for sen_ids, y, raw_sen, rational in doclist:
            start = 0
            for ind, w in enumerate(sen_ids):
                if w in split_tokens:
                    sen_list.append(sen_ids[start:ind + 1])
                    start = ind + 1

            if start < len(sen_ids) - 1:
                sen_list.append(sen_ids[start:])

        return sen_list

def tqdm_wrap(iterable, *args, **kwargs):
    """Forward an iterable eventually wrapped around a tqdm decorator
    The iterable is only wrapped if the iterable contains enough elements
    Args:
        iterable (list): An iterable object which define the __len__ method
        *args, **kwargs: the tqdm parameters
    Return:
        iter: The iterable eventually decorated
    """
    if len(iterable) > 100:
        return tqdm(iterable, *args, **kwargs)
    return iterable
