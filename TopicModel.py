#!/usr/bin/env python
# coding: utf-8



import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')




import nltk
nltk.download('stopwords')




from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])




COLUMNS=['doc_id','date','pres','title','speech']
df = pd.DataFrame(columns=COLUMNS)
PRES = 'obama'
pres_list = "adams arthur bharrison buchanan bush carter cleveland clinton coolidge eisenhower fdroosevelt fillmore ford garfield grant gwbush harding harrison hayes hoover jackson jefferson johnson jqadams kennedy lbjohnson lincoln madison mckinley monroe nixon obama pierce polk reagan roosevelt taft taylor truman tyler vanburen washington wilson".split()




import os
import re
_id = 1
for filename in os.listdir('./speeches'):
    if filename == '.DS_Store':
        continue
    for speech in os.listdir('./speeches/' + filename):
        temp = open('./speeches/' + filename + '/' + speech, 'r', encoding='utf-8').readlines()
        obj = {}
        obj['doc_id'] = _id
        date = re.findall('"([^"]*)"', temp[1])
        obj['date'] = date[0] if len(date) > 0 else None
        obj['pres'] = filename
        obj['title'] = re.findall('"([^"]*)"', temp[0])[0]
        obj['speech']= "".join(temp[2:])
    
        obj = pd.DataFrame(obj, index=[0])
        df = df.append(obj, ignore_index=True)
        _id += 1
df = df.set_index("doc_id")
df.head()




docs = df[df['pres'] == PRES]
docs.head()




data = docs.speech.values.tolist()
data = [re.sub('\n+', ' ', sent) for sent in data]
data = [re.sub(r"\<.*\>"," ",sent) for sent in data]




def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        
data_words = list(sent_to_words(data))

print(data_words[:1])




bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

print(trigram_mod[bigram_mod[data_words[0]]])




def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out




# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])




# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])


# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=5, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
lda_model.save('./models/'+PRES+'-model')


file = open("./output/"+PRES+".txt","w")
for topic in lda_model.print_topics():
    print(str(topic[0]) + ": " + topic[1])
    file.write(str(topic[0]) + ": " + topic[1])
file.close()
doc_lda = lda_model[corpus]


print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


#pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis
