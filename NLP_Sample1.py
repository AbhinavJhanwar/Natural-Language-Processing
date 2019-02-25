'''
Created on Feb 9, 2018

@author: abhinav.jhanwar
'''

''' USING RIDGE ALGORITHM & COUNTVECTORIZER'''

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

# define input
#text = open("Tripadvisor_hotelreviews_Shivambansal.txt",'r',encoding='latin1').read().lower()
text = [
    "PretzelBros, airbnb for people who like pretzels, raises $2 million",
    "Top 10 reasons why Go is better than whatever language you use.",
    "Why working at apple stole my soul (I still love it though)",
    "80 things I think you should do immediately if you use python.",
    "Show HN: carjack.me -- Uber meets GTA"
    ]

submissions = pd.read_csv("stories.csv").dropna().head(9356)
submissions.columns = ['a', 'submission_time', 'b', 'c', 'd', 'url', 'upvotes', 'headline']
#print(submissions.columns.values.tolist())

# tokenizing sentencewise
#sentences = sent_tokenize(text, 'english')

# Construct a bag of words matrix.
## This will lowercase everything, and ignore all punctuation by default.
## It will also remove stop words.
vectorizer = CountVectorizer(lowercase=True, stop_words='english')
##matrix = vectorizer.fit_transform(text)

# Let's apply the same method to all the headlines in all 100000 submissions.
# We'll also add the url of the submission to the end of the headline so we can take it into account.
submissions['full_test'] = submissions['headline']+' '+submissions['url']
matrix = vectorizer.fit_transform(submissions['headline'])
#print(matrix.toarray())
#print(vectorizer.get_feature_names())
#print(matrix.shape)

# dimensionality reduction
## Convert the upvotes variable to binary so it works with a chi-squared test.
col = submissions['upvotes'].copy(deep=True)
col_mean = col.mean()
col[col<col_mean]=0
col[(col>0) & (col>col_mean)]=1
## Find the 1000 most informative columns
selector = SelectKBest(chi2, k=1000)
selector.fit(matrix, col)
top_words = selector.get_support().nonzero() 
## Pick only the most informative columns in the data.
chi_matrix = matrix[:,top_words[0]]
#print(chi_matrix.shape)

# Adding meta features
## Our list of functions to apply.
transform_functions=[
    lambda x: len(x),
    lambda x: x.count(" "),
    lambda x: x.count('.'),
    lambda x: x.count('!'),
    lambda x: x.count('?'),
    lambda x: len(x)/(x.count('.')+1),
    lambda x: len(re.findall('\d', x)),
    lambda x: len(re.findall('[A-Z]', x)),
]
## Apply each function and put the results into list.
columns = list()
for func in transform_functions:
    columns.append(submissions['headline'].apply(func))
## convert the meta features to a numpy array
meta = np.asarray(columns).T

# Adding more features
columns = list()
## Convert the submission dates column to datetime.
submission_dates = pd.to_datetime(submissions['submission_time'])
## Transform functions for the datetime column.
transform_functions = [
    lambda x: x.year,
    lambda x: x.month,
    lambda x: x.day,
    lambda x: x.hour,
    lambda x: x.minute,
]
## Apply all functions to the datetime column.
for func in transform_functions:
    columns.append(submission_dates.apply(func))
## Convert the meta features to a numpy array
non_nlp = np.asarray(columns).T
## Concatenate the features together.
features = np.hstack([non_nlp, meta, chi_matrix.todense()])

# Making Predictions
model = Ridge(alpha=.1)
X = features
y = submissions['upvotes']
score = np.sqrt((-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)).mean())
print(score)
reg = model.fit(X, y)

indices = list(range(features.shape[0]))
random.shuffle(indices)
# Create test sets.
X = features[indices[:20], :]

predictions = pd.DataFrame(data = {'PredictedValues':reg.predict(X)},index=y.head(20).index)
print(pd.concat([predictions, y.head(20)],axis=1))


'''def make_matrix(sentences, uniqueWords):
    # sentences = list of sentences
    matrix = []
    for sentence in sentences:
        # Count each word in the sentence, and make a dictionary.
        counter = Counter(sentence)
        # Turn the dictionary into a matrix row using the uniqueWords.
        row = [counter.get(w,0) for w in uniqueWords]
        matrix.append(row)
    df = pd.DataFrame(matrix)
    df.columns = uniqueWords
    return df
    
# convert into lower case
sentences = [line.lower() for line in text]
#print(sentences)

# remove all regular expressions & numbers
for i in range(0,len(sentences)):
    # remove regular expressions
    sentences[i] = re.sub(r'[-./?!,":;()\']',' ',sentences[i])
    # remove all numbers
    sentences[i] = re.sub('[-|0-9]',' ', sentences[i])

# remove all stop words
## get list of stop words
stopwordsList = stopwords.words('english')
# create new list to store words of sentences
sentencesWords = list(sentences)
for i in range(0,len(sentencesWords)):
    # tokenize for words
    sentencesWords[i] = word_tokenize(sentencesWords[i], 'english')
    sentencesWords[i] = [word for word in sentencesWords[i] if word not in stopwordsList]

# extract unique words from sentences
textHeaders = set()
for i in range(0,len(sentencesWords)):
    textHeaders.update(sentencesWords[i])

data = make_matrix(sentencesWords, textHeaders)'''




#print(data)        
# getting pos tags of all the words
#all_tags = pos_tag(sentences[0])
#print(all_tags)
