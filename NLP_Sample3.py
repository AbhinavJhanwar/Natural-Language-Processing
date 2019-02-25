'''
Created on Feb 9, 2018

@author: abhinav.jhanwar
'''

import re
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import Ridge
import random
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
tqdm.pandas(tqdm())

# define input
#text = open("Tripadvisor_hotelreviews_Shivambansal.txt",'r',encoding='latin1').read().lower()
text = [
    "PretzelBros, airbnb for people who like pretzels, raises $2 million",
    "Top 10 reasons why Go is better than whatever language you use.",
    "Why working at apple stole my soul (I still love it though)",
    "80 things I think you should do immediately if you use python.",
    "Show HN: carjack.me -- Uber meets GTA"
    ]

df = pd.DataFrame({"comments":text, "upp":text})

def convertPosTag(pos_tuple):
    if pos_tuple[1][0].lower() == 'n':
        return 'n'
    elif pos_tuple[1][0].lower() == 'v':
        return 'v'
    elif pos_tuple[1][0].lower() == 'j':
        return 'a'
    elif pos_tuple[1][0].lower() == 'r':
        return 'r'
    else:
        return 'n'

def processText(sentence, regularExpression, lemmetize, stem):
    #######################################
    #
    # input: sentence
    # output: convert into lower case, remove regular expressions
    #
    #######################################
    print(sentence)
    ###############check for stop words as well if word is not in stopwords then only apply lemmetize or stem # to be applied
    data = sentence.lower()
    
    # regular Expression is true then process otherwise skip
    if regularExpression:
        # remove all regular expressions & numbers
        data = data.progress_apply(lambda sentence:
                                   re.sub(r'[-./?!,":;()\']', ' ', sentence))
        data = data.progress_apply(lambda sentence:
                                   re.sub('[-|0-9]', ' ', sentence))
    
    if lemmetize:
        lemmatizer = WordNetLemmatizer()
        data = data.progress_apply(lambda sentence: 
                                   ' '.join([lemmatizer.lemmatize(word[0], convertPosTag(word)) for word in pos_tag(word_tokenize(sentence, 'english'))]))
    if stem:
        stemmer = PorterStemmer()
        data = data.progress_apply(lambda sentence:
                                   ' '.join([stemmer.stem(word) if wordnet.synsets(stemmer.stem(word)) else word for word in word_tokenize(sentence, 'english')]))
    return data

#print(df)
df['comments'] = processText(df['comments'], regularExpression=True, lemmetize=True, stem=True)
# lemmatization
print(df)

# Construct a bag of words matrix.
vectorizer = CountVectorizer(ngram_range=(1,3), lowercase=True, stop_words='english', max_df=0.99, min_df=0.02)
matrix = vectorizer.fit_transform(text)
#print(matrix.toarray())
#print(vectorizer.get_feature_names())
#print(matrix.shape)

# create dataframe of converted data
data = pd.DataFrame(data=matrix.toarray(), columns=vectorizer.get_feature_names())
#print(data)