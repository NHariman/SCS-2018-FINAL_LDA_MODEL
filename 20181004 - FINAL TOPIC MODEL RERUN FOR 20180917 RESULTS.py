"""
Created on 04 10 2018

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

#load data containing documents that have been lemmatized and do not contain stopwords, special characters or documents with less than 20 words in python LIST format
#list has two rows, one with the document and the other with the date of said document
with open(('FILE.py'), 'rb') as fp:
            no_stops_sentences = pickle.load(fp)

print("loaded data")

#shuffle sentences
shuffle_sentences = list(no_stops_sentences)  
shuffle(shuffle_sentences)

print("shuffled")

print(shuffle_sentences[0])


count_list = []
for i in range(len(shuffle_sentences)):
    count_list.append(shuffle_sentences[i][0])
print(count_list[0])

#%%

data = count_list
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

#create bigrams and trigrams
"""build bigram"""
bigram = gensim.models.Phrases(data_words, min_count=20, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
print("loaded")

data_words_bigrams = make_bigrams(data_words)  
data_words_trigrams = make_trigrams(data_words)
print("created bigrams and trigrams")

#save all trigrams
trigram_list = []
for i in range(len(data_words_trigrams)):
    for j in range(len(data_words_trigrams[i])):
        if "_" in data_words_trigrams[i][j]:
            trigram_list.append(data_words_trigrams[i][j])
nodub_trigram_list = list(set(trigram_list))

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[1])
len(data_lemmatized)

#calculate median length of documents
median_length = []
for i in range(len(data_lemmatized)):
    median_length.append(float(len(data_lemmatized[i])))
    
print("Average Document Word Length")
print("median: ", statistics.median(median_length)) #261.0
print("mean: ", statistics.mean(median_length)) #239.29680150517405
print("minimum: ", min(median_length)) #0.0
print("maximum: ", max(median_length)) #2238.0

np_array = np.array(median_length)
print("25% percentile: ", np.percentile(np_array,25))
print("50% percentile: ", np.percentile(np_array,50)) 
print("75% percentile: ", np.percentile(np_array,75))

#observations: texts are VERY SHORT
#Average Document Word Length
#median:  25.0
#mean:  25.17629804745644
#minimum:  9.0
#maximum:  188.0
#25% percentile:  22.0
#50% percentile:  25.0
#75% percentile:  28.0

lem_sentences = []
for i in range(len(data_lemmatized)):
    lem_sentences.append(" ".join(data_lemmatized[i]))
print(lem_sentences[0])


#count vectorize, remove words that occur less than 10 times or in more than 25% of documents and contain less than 3 characters (letters)
tf_vectorizer = CountVectorizer(min_df=10, max_df=0.25,token_pattern='[a-zA-Z0-9]{3,}')
tf = tf_vectorizer.fit_transform(lem_sentences)
terms_dataframe = DataFrame(tf.A, columns=tf_vectorizer.get_feature_names())
terms = list(terms_dataframe)

w, h = 2, len(terms)
term_list = [[0 for x in range(w)] for y in range(h)]
for i in range(len(terms)):
    term_list[i][0] = terms[i]
    term_list[i][1] = terms_dataframe[terms[i]].sum()


term_values = []
for i in range(len(term_list)):
    term_values.append(term_list[i][1])
#%%
print("median:", statistics.median(term_values))
np_array = np.array(term_values)
print("25% percentage:", np.percentile(np_array,25)) #17
print("50% percentage:", np.percentile(np_array,50)) #31
print("75% percentage:", np.percentile(np_array,75)) #83.0

#based on 25% percentile, remove any term that occurs less than 17 times within the entire corpus
new_term_list = []
skipped_terms = []
for i in range(len(term_list)):
    if term_list[i][1] < 17:
        print("skipped: ", term_list[i])
        skipped_terms.append(term_list[i])
    else:
        print("added: ", term_list[i])
        new_term_list.append(term_list[i])

#check which terms are related to the main 8 search terms
search_terms = ["cancer","chemotherapy","tumor","carcinogen","lump","malignant","mutation","screening","therapy"]
print(search_terms)
search_terms_list =[]
for j in range(len(search_terms)):
    for i in range(len(term_list)):
        if search_terms[j] in term_list[i][0]:
            search_terms_list.append(term_list[i][0])
print(search_terms_list)
#add immunotherapy and cancercausing back to list

terms_solo = []
for i in range(len(new_term_list)):
    terms_solo.append(new_term_list[i][0])


#check if word is in the list
for i in range(len(search_terms_list)):
    print(search_terms_list[i], " in terms_list: ", search_terms_list[i] in terms_solo)


#add search terms that aren't in current list back to that list
new_terms = list(terms_solo)
print(len(search_terms_list))
for i in range(len(search_terms_list)):
    if search_terms_list[i] not in terms_solo:
        new_terms.append(search_terms_list[i])
        print("added:", search_terms_list[i])


#add trigrams back into accepted terms: trigram_list
new_trigram_terms = list(new_terms)
print(len(search_terms_list))
for i in range(len(new_trigram_terms)):
    if trigram_list[i] not in new_terms:
        new_terms.append(trigram_list[i])
        print("added:", trigram_list[i])



#remove words that are not in the new accepted term list
n_data = list(lem_sentences)
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

lem_data_words = list(sent_to_words(n_data))

print(lem_data_words[:1])

#accepted terms: new_terms
#documents: lem_data_words
data_filtered = []
for i in range(len(lem_data_words)):
    snippet = lem_data_words[i]
    filtered_snippet = []
    for j in range(len(lem_data_words[i])):
        if snippet[j] in new_terms:
            filtered_snippet.append(snippet[j])
    data_filtered.append(filtered_snippet)

#lemmetize again
#data_lemmatized_v2: terms that occured less than 17 times and occured more than 25% in the corpus filtered out


# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized_v2 = lemmatization(data_filtered, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

#save results
with open(("FILE.py"), 'wb') as fp:
    pickle.dump(data_lemmatized_v2, fp)


#create train and test dataset
train_set = data_lemmatized_v2[:30000]
test_set = data_lemmatized_v2[30000:]
id2word = corpora.Dictionary(data_lemmatized_v2) 

texts = list(train_set)

corpus = [id2word.doc2bow(text) for text in texts]

texts2 = list(test_set)
corpus2 = [id2word.doc2bow(text) for text in texts2]

print(corpus[:1])
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

text_full = list(data_lemmatized_v2)
final_corpus = [id2word.doc2bow(text) for text in text_full]

print(final_corpus[:1])
[[(id2word[id], freq) for id, freq in cp] for cp in final_corpus[:1]]

print("corpus1")
# Human readable format of corpus (term-frequency)
print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])
# Human readable format of corpus (term-frequency)
print("corpus2")
print([[(id2word[id], freq) for id, freq in cp] for cp in corpus2[:1]])
# Human readable format of corpus (term-frequency)
print("last corpus")
print([[(id2word[id], freq) for id, freq in cp] for cp in final_corpus[:1]])

"""
evaluate various LDA models with different topic amounts:
5, 10, 15, 20, 25
"""
#5 topics model
num_topics=5
lda_model = gensim.models.ldamodel.LdaModel(corpus=final_corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           alpha=1.0/num_topics, #sparsity
                                           update_every=1,
                                           chunksize=1000,
                                           passes=10,
                                           per_word_topics=True)

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, final_corpus, id2word)
vis

#10 topics model
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


#15 topics model
num_topics=15
lda_model3 = gensim.models.ldamodel.LdaModel(corpus=final_corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           alpha=1.0/num_topics, #sparsity
                                           update_every=1,
                                           chunksize=1000,
                                           passes=10,
                                           per_word_topics=True)

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model3, final_corpus, id2word)
vis


#20 topics model
num_topics=20
lda_model4 = gensim.models.ldamodel.LdaModel(corpus=final_corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           alpha=1.0/num_topics, #sparsity
                                           update_every=1,
                                           chunksize=1000,
                                           passes=10,
                                           per_word_topics=True)

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model4, final_corpus, id2word)
vis


#25 topics model
num_topics=25
lda_model5 = gensim.models.ldamodel.LdaModel(corpus=final_corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           alpha=1.0/num_topics, #sparsity
                                           update_every=1,
                                           chunksize=1000,
                                           passes=10,
                                           per_word_topics=True)

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model5, final_corpus, id2word)
vis
