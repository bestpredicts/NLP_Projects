import torch
from torch import nn
import gensim
import fasttext
from gensim.models import FastText
import random
import operator


model_dir = "../clinicalBERT/word2vec+fastText/word2vec+fastText"
model = gensim.models.KeyedVectors.load(f'{model_dir}/word2vec.model')


# def get_word_

#load in gensim format
weights = model.wv

# convert to tensor for pytorch
weights = torch.FloatTensor(model.wv.vectors)
print(weights.shape)
print(weights)
# convert to embedding layer
embedding = nn.Embedding.from_pretrained(weights)

# print(embedding)
#
# # model_fast =FastText.load_fasttext_format(f'{model_dir}/FastTExt/fasttext.model')
#
# words = model.wv.index_to_key
# indexes = model.wv.key_to_index
# print(words)
# print(indexes)
# print(indexes["patient"])
#
# # sorted_d = sorted(indexes.items(), key=operator.itemgetter(1))
# # print('Dictionary in ascending order by value : ',sorted_d)
# # print(indexes["START_TOKEN"])
#
# # print(len(words))
#
# random_word = random.choice(model.wv.index_to_key)
# # print(random_word)

#
#
# patient_idx = model.wv.key_to_index["patient"]
# print(patient_idx)
# patient_cnt = model.wv.get_vecattr("patient", "count")  # 👍
# print(patient_cnt)
# vocab_len = len(model.wv)
# print(vocab_len)
#
# print(model.wv.most_similar("patient"))