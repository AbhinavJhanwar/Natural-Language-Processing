# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 12:09:29 2019

@author: abhinav.jhanwar
"""

# Python program to generate word vectors using Word2Vec 
  
# importing all necessary modules 
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
  
warnings.filterwarnings(action = 'ignore') 
  
import gensim 
from gensim.models import Word2Vec 

from sklearn.decomposition import PCA
from matplotlib import pyplot

###################################################### SAMPLE 1 ############################
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]

# Create CBOW model 
# size: (default 100) The number of dimensions of the embedding, e.g. the length of the dense vector to represent each token (word).
# window: (default 5) The maximum distance between a target word and words around the target word.
# min_count: (default 5) The minimum count of words to consider when training the model; words with an occurrence less than this count will be ignored.
# workers: (default 3) The number of threads to use while training.
# sg: (default 0 or CBOW) The training algorithm, either CBOW (0) or skip gram (1).
model = Word2Vec(sentences, min_count=1, size=50, window=6, workers=12)

# summarize the loaded model
print(model)

# summarize vocabulary
words = list(model.wv.vocab)
print(words)

# access vector for one word
print(model['sentence'])

# save model
model.save('model.bin')

# load model
new_model = Word2Vec.load('model.bin')
print(new_model)

# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

model.wv.similarity('first', 'second')























########################################################### SAMPLE 2 ############################  
#  Reads ‘alice.txt’ file 
sample = open("alice.txt", "r") 
s = sample.read() 
  
# Replaces escape character with space 
f = s.replace("\n", " ") 
  
data = [] 
  
# iterate through each sentence in the file 
for i in sent_tokenize(f): 
    temp = [] 
      
    # tokenize the sentence into words 
    for j in word_tokenize(i): 
        temp.append(j.lower()) 
  
    data.append(temp) 
model1 = gensim.models.Word2Vec(data, min_count = 1,  
                              size = 100, window = 5, workers=12) 
  
# Print results 
print("Cosine similarity between 'alice' " + 
               "and 'wonderland' - CBOW : ", 
    model1.similarity('alice', 'wonderland')) 
      
print("Cosine similarity between 'alice' " +
                 "and 'machines' - CBOW : ", 
      model1.similarity('alice', 'machines')) 
  
# Create Skip Gram model 
model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100, 
                                             window = 5, workers=12, sg = 1) 
  
# Print results 
print("Cosine similarity between 'alice' " +
          "and 'wonderland' - Skip Gram : ", 
    model2.similarity('alice', 'wonderland')) 
      
print("Cosine similarity between 'alice' " +
            "and 'machines' - Skip Gram : ", 
      model2.similarity('alice', 'machines')) 