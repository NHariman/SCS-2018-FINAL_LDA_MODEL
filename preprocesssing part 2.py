# -*- coding: utf-8 -*-
"""
Created on Fri May 25 14:13:41 2018

@author: Naomi Hariman
"""
#%%
"""
PART ONE: CONSTRUCTING A VOCABULARY
"""
#load packages
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
#open list of documents that do not contain any uppercase characters, special characters or short documents
with open(('FILE.py'), 'rb') as fp:
            raw_data = pickle.load(fp)        
#tokenize
w, h = 2, len(raw_data);
tokens = [[0 for x in range(w)] for y in range(h)]
for i in range(len(raw_data)):
    tokens[i][0] = nltk.word_tokenize(raw_data[i][0])
    tokens[i][1] = str(raw_data[i][1])
    print("finished document: ", i)
#stop word removal
stops = stopwords.words('english')
w, h = 2, len(stops);
stops_new = [] 
for i in range(len(stops)):
    stops_new.append(re.sub('[^a-zA-Z ]+', ' ', stops[i]))
#%%
stops_sentence = " ".join(stops_new)

stop_list = stops_sentence.split()
#create list of stopwords
seen = set()
stop_tokens = []
for item in stop_list:
    if item not in seen:
        seen.add(item)
        stop_tokens.append(item)
#remove stopwords
w, h = 2, len(tokens);
remove_stops = [[0 for x in range(w)] for y in range(h)]
for i in range(len(tokens)):  
    remove_stops[i][0] = [w for w in tokens[i][0] if not w in stop_tokens]
    remove_stops[i][1] = str(tokens[i][1])
#remove short documents after stopword removal
no_short_strings = []
for i in range(len(remove_stops)):
    if len(remove_stops[i][0]) < 20:
        print("skipped document: ", i)
    else:
        no_short_strings.append(remove_stops[i])
len(no_short_strings)
#%%
w, h = 2, len(no_short_strings);
no_stops_sentences = [[0 for x in range(w)] for y in range(h)]
for i in range(len(no_short_strings)):
    no_stops_sentences[i][0] = " ".join(no_short_strings[i][0])
    no_stops_sentences[i][1] = str(no_short_strings[i][1])
#save file
with open(("FILE.py"), 'wb') as fp:
    pickle.dump(no_stops_sentences, fp)