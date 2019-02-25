# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 12:51:39 2018

@author: abhinav.jhanwar
"""

from processText import text_process
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

data = pd.read_csv('jokes.csv').dropna()
data_text = data[['Raw_joke']]
data_text['index'] = data_text.index
documents = data_text

processed_docs = documents['Raw_joke'].progress_apply(text_process, pos=True, lemmetize=True, stem=True)
print(processed_docs[:10])

vectorizer = CountVectorizer(ngram_range=(1,2), stop_words='english', lowercase=True, max_df=0.5, min_df=0.02)
bow_corpus = vectorizer.fit_transform()