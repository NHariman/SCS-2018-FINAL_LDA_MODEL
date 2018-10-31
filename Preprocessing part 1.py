# -*- coding: utf-8 -*-
"""
Created on Thu May 24 13:43:33 2018

@author: Naomi Hariman
"""
#load libraries
import pickle
import re
import statistics
import nltk
import string
from nltk.stem.porter import *
from nltk.corpus import stopwords
import os
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import scipy.sparse
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from __future__ import print_function

from pandas import DataFrame
#open list of documents
with open(('FILE'), 'rb') as fp:
            all_data = pickle.load(fp)
#turn into matrix, separate date from header and snippet
w, h = 2, len(all_data);
date_matrix = [[0 for x in range(w)] for y in range(h)] 
for i in range(len(all_data)):
    date_matrix[i][0] = all_data[i][11:]
    date_matrix[i][1] = all_data[i][:8]
#remove special characters + numbers, replaces all special characters with a space
w, h = 2, len(date_matrix);
no_numbers = [[0 for x in range(w)] for y in range(h)] 
for i in range(len(date_matrix)):
    no_numbers[i][0] = re.sub('[^a-zA-Z ]+', ' ', date_matrix[i][0])
    no_numbers[i][1] = str(date_matrix[i][1])
#%%
remove double spaces and special white spaces    
w, h = 2, len(date_matrix);
no_ws = [[0 for x in range(w)] for y in range(h)]
for i in range(len(no_numbers)):
    no_ws[i][0] = " ".join(no_numbers[i][0].split())
    no_ws[i][1] = str(no_numbers[i][1])

#%%Tokenize (get wordcount)

w, h = 2, len(no_ws);
tokens = [[0 for x in range(w)] for y in range(h)]
#tokens = {}
for i in range(len(no_ws)):
    tokens[i][0] = nltk.word_tokenize(no_ws[i][0])
    tokens[i][1] = str(no_ws[i][1])
    print("finished document: ", i)
#%%check for median?
median_length = []
for i in range(len(tokens)):
    median_length.append(float(len(tokens[i][0])))

print("Average Document Word Length")
print("median: ", statistics.median(median_length)) #261.0
print("mean: ", statistics.mean(median_length)) #239.29680150517405
print("minimum: ", min(median_length)) #0.0
print("maximum: ", max(median_length)) #2238.0

np_array = np.array(median_length)
print("25% percentile: ", np.percentile(np_array,25)) #196.0
print("50% percentile: ", np.percentile(np_array,50)) #261.0
print("75% percentile: ", np.percentile(np_array,75)) #286.0
#remove short documents to prevent interference
#removed documents with 20 words or less based on 
#http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf
no_short_strings = []
for i in range(len(tokens)):
    if len(tokens[i][0]) < 20:
        print("skipped document: ", i)
    else:
        no_short_strings.append(tokens[i])
#turn tokens back into sentences
w, h = 2, len(no_short_strings);
data_sentences = [[0 for x in range(w)] for y in range(h)]
for i in range(len(no_short_strings)):
    data_sentences[i][0] = " ".join(no_short_strings[i][0])
    data_sentences[i][1] = str(no_short_strings[i][1])
#convert text to lowercase
w, h = 2, len(data_sentences);
processed_data = [[0 for x in range(w)] for y in range(h)] 
for i in range(len(data_sentences)):
    processed_data[i][0] = data_sentences[i][0].lower()
    processed_data[i][1] = str(data_sentences[i][1])
#create matrix
Matrix_processed = np.array(processed_data)
#save files
np.savetxt('FILE.txt', Matrix_processed, delimiter=',', fmt='%s')
with open(("FILE.py"), 'wb') as fp:
    pickle.dump(processed_data, fp)
#%%

