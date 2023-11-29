# %%
from nltk.tag import pos_tag
import nltk
import spacy
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load("en_core_web_sm")


# %%
text = [
    "PretzelBros, airbnb for people who like pretzels, raises $2 million",
    "Top 10 reasons why Go is better than whatever language you use.",
    "Why working at apple stole my soul (I still love it though)",
    "80 things I think you should do immediately if you use python.",
    "Show HN: carjack.me -- Uber meets GTA",
    "Apple is looking at buying U.K. startup for $1 billion"
    ]
doc = nlp(text[0])

# %%
# POS TAGGING
# pos tagging in spacy
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_)
    
words = nltk.word_tokenize(text[5])  
# Part of speech Tagging in nltk               
tag_words = pos_tag(words)
for tag_word in tag_words:
    print(tag_word)

# %%
print(doc)
# NAMED ENTITY RECOGNITION
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
    