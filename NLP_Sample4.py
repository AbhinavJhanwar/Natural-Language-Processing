'''
Created on Feb 26, 2018

@author: abhinav.jhanwar
'''

''' using jokes dataset'''

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.pipeline import Pipeline
from tqdm import tqdm   # to check progress of for loop
#tqdm.pandas(tqdm())

df = pd.read_json("https://raw.githubusercontent.com/drvinceknight/EdinburghFringeJokes/master/jokes.json")

df.Year = df.Year.apply(int)

commonWords = stopwords.words('english')
commonWords.extend(['m', 've'])

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

df.Raw_joke = df.Raw_joke.apply(lambda sentence:
                                [sent.lower() for sent in tokenizer.tokenize(sentence) if sent.lower() not in commonWords])

all_words = [item for sublist in [word for word in df[df.Year<=2013].Raw_joke] for item in sublist]

all_words = set(all_words)

def extract_features(joke, all_words):
    words = set(joke)
    features = {}
    for word in words:
        features[word] = (word in all_words)
    return features

df['Features'] = df.Raw_joke.apply(lambda sentence:
                                extract_features(sentence, all_words))

funny_threshold=5
df.Rank = df.Rank.apply(int)
df['Funny'] = df.Rank<=funny_threshold
df['LabeledFeature'] = list(zip(df.Features, df.Funny))

classifier = nltk.NaiveBayesClassifier.train(df[df.Year<=2013].LabeledFeature)

#classifier.show_most_informative_features(10)

joke = "Clowns divorce. Custardy battle"
print(classifier.classify(extract_features([sent.lower() for sent in tokenizer.tokenize(joke) if sent.lower() not in commonWords], all_words)))

#print(features)
#print(df.head())



