# -*- coding: utf-8 -*-
"""
@author: Naomi Hariman
"""
'''
TO DO LIST
- split documents by year. (every 10 years)
   1 -1970-1979
   2 -1980-1989
   3 -1990-1999
   4 -2000-2009
   5 -2010-2018
- apply preprocessing to all 5 documents
- apply model to all 5 documents

TO INSPECT:
- trends that occur over the years (continuous)
- trends that are specific to a certain time frame (temporal)
- topics with a strong temporal occurance (see Jacobi paper, figure 3
- find most relevant document per topic?
'''
#LOAD LIBRARIES
import pickle
import re
import statistics
import nltk
import string
from nltk.stem.porter import *
from nltk.corpus import stopwords
import os
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from __future__ import print_function

from nltk.stem.porter import PorterStemmer
import scipy.sparse
import numpy as np
from pandas import DataFrame 
from random import shuffle   

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt


# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

print("imported packages")

#LOAD DATA
with open(('FILE'), 'rb') as fp:
            data_lemmatized_dct_v3 = pickle.load(fp)
			
#separate per 10 years
'''
   1 -1970-1979
   2 -1980-1989
   3 -1990-1999
   4 -2000-2009
   5 -2010-2018

print(no_stops_sentences[1][1][0:4])
'''
year70 = []
for i in range(10):
    year70.append(str(1970+i))
year80 = []
for i in range(10):
    year80.append(str(1980+i))
year90 = []
for i in range(10):
    year90.append(str(1990+i))
year00 = []
for i in range(10):
    year00.append(str(2000+i))
year10 = []
for i in range(9):
    year10.append(str(2010+i))

data70=[]
data80=[]
data90=[]
data00=[]
data10=[]
for i in range(len(data_lemmatized_dct_v3)):
    if data_lemmatized_dct_v3[i][0][2] in year70:
        data70.append(data_lemmatized_dct_v3[i])
    if data_lemmatized_dct_v3[i][0][2] in year80:
        data80.append(data_lemmatized_dct_v3[i])
    if data_lemmatized_dct_v3[i][0][2] in year90:
        data90.append(data_lemmatized_dct_v3[i])
    if data_lemmatized_dct_v3[i][0][2] in year00:
        data00.append(data_lemmatized_dct_v3[i])
    if data_lemmatized_dct_v3[i][0][2] in year10:
        data10.append(data_lemmatized_dct_v3[i])

yeardata = [data70, data80, data90, data00, data10]

#create dictionary for all the count_list
x = ["70","80", "90", "00", "10"]
year_dct = {}
for i in x:
    year_dct[i] = []
yearkeys = list(year_dct.keys())
print(yearkeys)

for i in range(len(yeardata)):
    for j in range(len(yeardata[i])):
        year_dct[yearkeys[i]].append(yeardata[i][j][0][0])

#shuffle documents per time period
shuffle_70 = []
shuffle_80 = []
shuffle_90 = []
shuffle_00 = []
shuffle_10 = []

for i in range(len(year_dct["70"])):
    shuffle_70.append(year_dct["70"][i])
for i in range(len(year_dct["80"])):
    shuffle_80.append(year_dct["80"][i])
for i in range(len(year_dct["90"])):
    shuffle_90.append(year_dct["90"][i])
for i in range(len(year_dct["00"])):
    shuffle_00.append(year_dct["00"][i])
for i in range(len(year_dct["10"])):
    shuffle_10.append(year_dct["10"][i])
    

data_70 = list(shuffle_70)
shuffle(data_70)
data_80 = list(shuffle_80)
shuffle(data_80)
data_90 = list(shuffle_90)
shuffle(data_90)
data_00 = list(shuffle_00)
shuffle(data_00)
data_10 = list(shuffle_10)
shuffle(data_10)

#open non split data
with open(('FILE'), 'rb') as fp:
            data_lemmatized_v2 = pickle.load(fp)


id2word = corpora.Dictionary(data_lemmatized_v2) #not filtered like before, test run


text_full = list(data_lemmatized_v2)
final_corpus = [id2word.doc2bow(text) for text in text_full]

print(final_corpus[:1])
[[(id2word[id], freq) for id, freq in cp] for cp in final_corpus[:1]]

print("///////////////////CORPUS 70 ///////////////////////")
corpus70 = [id2word.doc2bow(text) for text in data_70]
print(corpus70[:1])
[[(id2word[id], freq) for id, freq in cp] for cp in corpus70[:1]]

print("///////////////////CORPUS 80 ///////////////////////")
corpus80 = [id2word.doc2bow(text) for text in data_80]
print(corpus80[:1])
[[(id2word[id], freq) for id, freq in cp] for cp in corpus80[:1]]

print("///////////////////CORPUS 90 ///////////////////////")
corpus90 = [id2word.doc2bow(text) for text in data_90]
print(corpus90[:1])
[[(id2word[id], freq) for id, freq in cp] for cp in corpus90[:1]]

print("///////////////////CORPUS 00 ///////////////////////")
corpus00 = [id2word.doc2bow(text) for text in data_00]
print(corpus00[:1])
[[(id2word[id], freq) for id, freq in cp] for cp in corpus00[:1]]

print("///////////////////CORPUS 10 ///////////////////////")
corpus10 = [id2word.doc2bow(text) for text in data_10]
print(corpus10[:1])
[[(id2word[id], freq) for id, freq in cp] for cp in corpus10[:1]]

#LDA MODEL ALL DOCUMENTS
num_topics=10
lda_model2 = gensim.models.ldamodel.LdaModel(corpus=final_corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           alpha=1.0/num_topics, #sparsity
                                           update_every=1,
                                           chunksize=1000,
                                           passes=10,
                                           per_word_topics=True)

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model2, final_corpus, id2word)
vis

#LDA MODEL 1970-1979

pyLDAvis.enable_notebook()
vis70 = pyLDAvis.gensim.prepare(lda_model2, corpus70, id2word)
print("////////////////////////////1970-1979//////////////////////////////////")
vis70

#LDA MODEL 1980-1989

pyLDAvis.enable_notebook()
vis80 = pyLDAvis.gensim.prepare(lda_model2, corpus80, id2word)

print("////////////////////////////1980-1989//////////////////////////////////")
vis80

#LDA MODEL 1990-1999

pyLDAvis.enable_notebook()
vis90 = pyLDAvis.gensim.prepare(lda_model2, corpus90, id2word)

print("////////////////////////////1990-1999//////////////////////////////////")
vis90

#LDA MODEL 2000-2009

pyLDAvis.enable_notebook()
vis00 = pyLDAvis.gensim.prepare(lda_model2, corpus00, id2word)

print("////////////////////////////2000-2009//////////////////////////////////")
vis00

#LDA MODEL 2010-2018

pyLDAvis.enable_notebook()
vis10 = pyLDAvis.gensim.prepare(lda_model2, corpus10, id2word)

print("////////////////////////////2010-2018//////////////////////////////////")
vis10