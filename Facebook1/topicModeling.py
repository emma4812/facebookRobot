# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 03:03:11 2022

@author: Emma
"""
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

# Create a corpus from a list of texts
#common_dictionary = Dictionary(common_texts)
#common_dictionary = Dictionary(documents)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

# Train the model on the corpus.
lda = LdaModel(common_corpus, num_topics=10)

text="the are killing going on in kawo right now"
t=common_dictionary.doc2bow(text.split())




import pandas as pd
data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False);
data_text = data[['headline_text']]
data_text['index'] = data_text.index
documents = data_text

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
#from nltk.stem.porter import PorterStemmer as stemmer
#import nltk.stemmer as stemmer
from nltk.stem import PorterStemmer
import numpy as np
np.random.seed(2018)
import nltk 
#nltk.download('wordnet')
#nltk.download('omw-1.4')

ps = PorterStemmer()
def lemmatize_stemming(text):
    return ps.stem(word=WordNetLemmatizer().lemmatize(text, pos='v'))
    #return stemmer.stem(word=text)

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


doc_sample = documents[documents['index'] == 4310].values[0][0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))


processed_docs = documents['headline_text'].map(preprocess)
processed_docs[:10]
dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
    
    
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)



bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[4310]




bow_doc_4310 = bow_corpus[4310]
for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 
                                               dictionary[bow_doc_4310[i][0]], 
bow_doc_4310[i][1]))
    
    
    
    
from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break


lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)



for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
    


lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=2)

for index, score in sorted(lda_model_tfidf[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))



unseen_document = 'he killed his son'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))
    
    lda_model_tfidf.get_topic_terms(lda_model_tfidf.top_topics())
 
    
    
from gensim.test.utils import datapath


# Save model to disk.
temp_file = datapath(r"C:\Users\Emma\Documents\UiPath\Facebook1\model")
lda_model.save(temp_file)

# Load a potentially pretrained model from disk.
lda = LdaModel.load(temp_file)


import pickle

def save_pickle(path,model):

    dbfile = open(path, 'ab')
      
    # source, destination
    pickle.dump(model, dbfile)                     
    dbfile.close()
  
def loadData(path):
    # for reading also binary mode is important
    dbfile = open(path, 'rb')     
    db = pickle.load('dictionary')
   
    dbfile.close()

path=r'C:\Users\Emma\Documents\UiPath\Facebook1\dictionary'
save_pickle(path, dictionary)



