'''
Created on Jan 29, 2018

@author: abhinav.jhanwar
'''

import re
import nltk
nltk.download_gui()
from nltk.corpus import stopwords
import wordcloud as wc
import matplotlib.pyplot as plt

data = {}
data['story'] = open('story.txt','r',encoding='utf-8').read()
#print(data['story'])

# convert all data in lower case
for k in data.keys():
    data[k] = data[k].lower()
#print(data['story'])

# remove all regular expressions
for k in data.keys():
    data[k] = re.sub(r'[-./?!,":;()\']',' ', data[k])
#print(data['story'])

# remove all numbers
for k in data.keys():
    data[k] = re.sub('[-|0-9]',' ', data[k])
#print(data['story'])

#print(stopwords.words('english'))
# get list of stop words
stopwords_list = stopwords.words('english')
#print(stopwords_list)

for k in data.keys():
    data[k] = data[k].split()
    
for k in data.keys():
    data[k] = [w for w in data[k] if w not in stopwords_list]

data['story'] = data['story']
#print(data['story'])

wordcloud = wc.WordCloud(width=1200, height=600).generate(' '.join(data['story']))
plt.figure(figsize=(17,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()