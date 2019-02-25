'''
Created on Jun 1, 2018

@author: abhinav.jhanwar
'''


import pandas as pd
import re
import nltk
#nltk.download_gui()
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import FreqDist

from textblob import TextBlob as tb

%matplotlib inline
import matplotlib.pyplot as plt

import urllib.request
from bs4 import BeautifulSoup
import math

# TEXT = ["Rose is sweet", "Goodnight sweet Prince", "Parting is sweet"]
# TF: Term Frequency, Number of time a word appears in a sentence = term instance/total terms
    # TF of "Sweet" in text1 = 1/3 = 0.33
# IDF: Inverse Document Frequency, Number of documents a word appears in them = log(number of docs/number of docs with term)
    # DF of sweet in text = log(3/3) = 0
# TF-IDF: multiplication of TF & IDF
    # TF-IDF for "sweet" = 0.33*0 = 0 hence sweet word is of less importance as it has high repeatation in most of the text 
    
    
def tf(word, doc):
    return doc.words.count(word)/len(doc.words)

def idf(word, docs):
    return math.log(len(docs)/(1+sum([1 for doc in docs if word in doc.words])))

def tfidf(word, doc, docs):
    return tf(word, doc)*idf(word, docs)

    
text = [
    "PretzelBros, airbnb for people who like pretzels, raises $2 million",
    "Top 10 reasons why Go is better than whatever language you use.",
    "Why working at apple stole my soul (I still love it though)",
    "80 things I think you should do immediately if you use python.",
    "Show HN: carjack.me -- Uber meets GTA"
    ]

doc1 = tb(text[0])
doc2 = tb(text[1])
doc3 = tb(text[2])
docs = [doc1, doc2, doc3]

for i, doc in enumerate(docs):
    print("top words in document{}".format(i+1))
    scores = {word: tfidf(word, doc, docs) for word in doc.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("\tWord: {}, tf-idf: {}".format(word, round(score, 5)))

# remove punctuation & digits
for i in range(len(text)):
    text[i] = re.sub('[-./?!,":;()\']',' ',text[i])
    text[i] = re.sub('[0-9]', ' ', text[i]).lower()

# tokenize words
words = word_tokenize(txt)
# remove stop words
words = [word for word in words if word not in stopwords]
# get frequency distribution of words
fdist = FreqDist(words)
# create data frame with index value 0 and transpose it
count_frame = pd.DataFrame(fdist, index=[0]).T
# define the index 0 column to 'Count'
count_frame.columns = ['Count']
# sort the data of dataframe in decreasing order
counts = count_frame.sort_values('Count', ascending=False)
# display top 50 words as bar plot
fig = plt.figure(figsize=(16,9))
ax = fig.gca()
counts['Count'][:50].plot(kind='bar', ax=ax)
plt.show()
