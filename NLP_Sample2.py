'''
Created on Feb 9, 2018

@author: abhinav.jhanwar
'''

'''USING RIDGE ALGORITHM'''
'''REGRESSION'''
'''using pipeline'''

import re
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestRegressor
import random
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm   # to check progress of for loop
tqdm.pandas(tqdm())

# define input
#text = open("Tripadvisor_hotelreviews_Shivambansal.txt",'r',encoding='latin1').read().lower()
submissions = pd.read_csv("C:/Users/abhinav.jhanwar/Downloads/stories.csv", encoding='latin1').dropna().head(50000)
submissions.columns = ['a', 'submission_time', 'b', 'c', 'd', 'url', 'upvotes', 'headline']
#print(submissions.columns.values.tolist())

submissions = pd.DataFrame({'upvotes':submissions.upvotes,'headline':submissions.headline})

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

def processText(data, regularExpression, lemmetize, stem):
    #######################################
    #
    # input: dataframe with list of sentences
    # output: convert into lower case, remove regular expressions
    #
    #######################################
    
    data = data.progress_apply(lambda sentence:
                               sentence.lower())
    
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

# process text  
submissions.headline = processText(submissions['headline'], regularExpression=False, lemmetize=True, stem=True)

# dimensionality reduction
## Find the 1000 most informative columns
selector = SelectKBest(chi2, k=1000)
#vectorizer = CountVectorizer(ngram_range=(1,3), max_df=0.99, min_df=0.02, lowercase=True, stop_words='english')
#matrix = vectorizer.fit_transform(submissions.headline)
#print(matrix.shape)
#17859
#170585

# define pipeline
text_pipeline = Pipeline([# min_df - it will be based on the target. i.e. if a target has category with minimum frequency of 5% then min_df will be approx 0.05 only
                          # max_df - it will be based on the target again i.e. sum of max frequencies of a top 5 or more or less target categories.
                          ('vect', CountVectorizer(ngram_range=(1,2), stop_words='english', lowercase=True, max_df=0.99, min_df=0.02)),
                          ('tfidf', TfidfTransformer()),
                          #('selectBest', selector),
                          #('model', Ridge(alpha=0.1))
                          ('model', RandomForestRegressor(n_estimators=180, min_samples_leaf=10))
                          #('model', LinearRegression())
                          ])

#print(text_pipeline.get_params(True))
# fitting model
text_pipeline = text_pipeline.fit(submissions.headline,submissions.upvotes)

# predicting model
predictions = text_pipeline.predict(submissions.headline.head())
#print(text_pipeline.score(submissions.headline, submissions.upvotes))

# getting rmse
score = np.sqrt((-cross_val_score(text_pipeline, submissions.headline, submissions.upvotes, scoring='neg_mean_squared_error', cv=5)).mean())
print(score)
