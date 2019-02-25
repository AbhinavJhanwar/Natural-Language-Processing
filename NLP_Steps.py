'''
Created on Feb 8, 2018

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

%matplotlib inline
import matplotlib.pyplot as plt

import urllib.request
from bs4 import BeautifulSoup




'''# TOKENIZATION OF WORDS FROM WEB PAGE
response = urllib.request.urlopen('http://php.net/')
html = response.read()
# remove html tags
soup = BeautifulSoup(html, "html5lib")
text = soup.get_text(strip=True)
# convert to lower case
text = text.lower()
# remove regular expressions
text = re.sub(r'[-./?!,":;()\']',' ',text)
# remove all numbers
text = re.sub('[-|0-9]',' ', text)
# generate tokens
tokens = text.split()
# work frequency
freq = nltk.FreqDist(tokens)
#for key,val in freq.items():
#    print(str(key)+":"+str(val))
freq.plot(20, cumulative=False)
# get list of stop words
stopwords_list = stopwords.words('english')
# remove stop words
cleaned_tokens = [w for w in tokens if w not in stopwords_list]
freq = nltk.FreqDist(cleaned_tokens)
freq.plot(20, cumulative=False)'''


'''# SENTENCE TOKENIZATION
myText = "Hello Adam, how are you? I hope everything is going well. Today is a good day, see you dude."
sentences = sent_tokenize(myText, 'english')
words = word_tokenize(myText, 'english')
# tokenizing non-English language
sentences = sent_tokenize("Bonjour M. Adam, comment allez-vous? J'espere que tout va bien. Aujourd'hui est un bon jour.", 'french')
# get synonyms of words
syn = wordnet.synsets("danger")
#print(syn[0].definition())
#print(syn[0].examples())
synonym = list()
for item in syn:
    for lemma in item.lemmas():
        synonym.append(lemma.name())
#print(synonym)

antonym = list()
for item in syn:
    for lemma in item.lemmas():
        if lemma.antonyms():
            antonym.append(lemma.antonyms()[0].name())
#print(antonym)'''


'''# WORD STEMMING
stemmer1 = PorterStemmer()
#print(stemmer1.stem('printing'))
# Lancaster stemming algorithm
stemmer2 = LancasterStemmer()
#print(stemmer2.stem('working'))
# using Snowball Stemmer for non-english languages
#print(SnowballStemmer.languages)
stemmer3 = SnowballStemmer('french')
#print(stemmer3.stem("Bonjour"))'''


# WORD LEMMATIZING
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("boy's", 'n'))
print(lemmatizer.lemmatize("boy's", 'v'))
print(lemmatizer.lemmatize("boy's", 'a'))
print(lemmatizer.lemmatize("boy's", 'r'))




















