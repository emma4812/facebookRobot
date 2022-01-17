# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 01:38:49 2022

@author: Emma
"""
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
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
import pickle
from gensim.test.utils import datapath
import re
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
def loadModel(path):
    temp_file = datapath(path+"/model_tfidf")
    lda = LdaModel.load(temp_file)
    return lda



def loadData(path):
    # for reading also binary mode is important
    dbfile = open(path+"/dictionary", 'rb')     
    dictionary = pickle.load(dbfile)
    dbfile.close()
    return dictionary

def isTopic(sentence):
    path=r'\Users\Emma\Documents\UiPath\Facebook1'
    dic=loadData(path)
    model=loadModel(path)
    bow_vector = dic.doc2bow(preprocess(sentence))
    i=0
    L=[]
    for index, score in sorted(model[bow_vector], key=lambda tup: -1*tup[1]):
        L= model.print_topic(index, 10).split('+')
        if i>0:
            break
        i+=1
        
    topics={}  
    for i in L:
        key=i.split("*")[1].replace('"', "").replace(' ', "")
        value=i.split("*")[0].replace('"', "").replace(' ', "")
        topics[key]=float( value)
        
    print(topics)    
    return topics

def keywordB(text):
    key1= [
      
      'Islamist', 'militant', 'Boko', 'Haram', 'defeated', 
      'north', 'east', 'security','conflict','deaths','deadly','IS',
      'deadliest','attacks','kidnapping','tension','army','bandits',
      'abductions','safe','Biafra','violating','gunshot','gun','shot',
      'injured','shooting','officer','kidnapped','kidnapper','killing',
      'killed','police', 'security', 'terms','Blood', 'Bloodshed',
      'Gunshot','Run', 'Afraid', 'Fast', 'Breakin', 'Break', 'Armed', 'Mask', 'Covered', 
      'Abduct', 'Shot', 'Outside', 'House', 'Home', 'Street', 'Hide', 'stole',
      'carry', 'fence', 'road', 'stab'
      ]
    return  len((set(key1).intersection(text.split(" "))))*0.04

def run1(text):
    

    security=[
       'restrict',
        'attack',
        'kill',
        'war',
        'dead',
        'live',
        'north',
        'south',
        'crash',
        'polic',
        'shoot',
        'die',
        'death',
        'murder',
        'lockdown',
        'case',
        'elect',
        'stab'
        
        ]
    
    
    sentences=re.split(';|, ',text)
    
    dic_topics=[]
    gw=0
    for sentence in sentences:

        topic= isTopic(str(sentence))
        x=set(security).intersection(set(topic.keys()))
    #t='''We are very saddened by this incessant invasion … and we also worried about the displaced persons who are fleeing in their hundreds from their communities,” Nneka Ikem Anibeze said on Sunday'''
        w=0
   
        for i in x:
            w+=topic[i]
        
        print(w+keywordB(text))
        gw+=w
        if w+keywordB(text)>0.04:
            return True
    
    if gw +keywordB(text)>0.04:
        return False
    return False
      


#run1()
t='''I just heard a gunshot close to my house at ungwan romi. i'm very scared'''
run1(t)




