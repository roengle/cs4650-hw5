#!/usr/bin/env python
# coding: utf-8

# In[26]:


import spacy as spacy

from newsapi.newsapi_client import NewsApiClient

from datetime import datetime
from dateutil.relativedelta import relativedelta

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')

from nltk import pos_tag
from nltk.corpus import wordnet 
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')


# In[27]:


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# In[29]:


nlp = spacy.load('en_core_web_lg')
newsapi = NewsApiClient (api_key='9514c8024c24490f8f2442464bfdaf81')

today = datetime.now()

today_str = today.strftime("%Y-%m-%d")
one_month_ago_str = (today - relativedelta(months=1) + relativedelta(days=1)).strftime("%Y-%m-%d")

words = []

for i in range(1,6):
    data = newsapi.get_everything(q='coronavirus', language='en', from_param='2022-02-28', to='2020-03-27', sort_by='relevancy', page=i)

    articles = data['articles']

    for index, article in enumerate(articles):
        publisher = article['source']['name']
        author = article['author']
        title = article['title']
        description = article['description']
        content = article['content']

        words.extend(nlp(title))
        words.extend(nlp(description))
        words.extend(nlp(content))

words = [w.text.lower() for w in words]
words = [w for w in words if w.isalnum()]
words = [w for w in words if w not in stopwords.words('english')]

words = [WordNetLemmatizer().lemmatize(w, get_wordnet_pos(w)) for w in words]


# In[30]:


# import wordcloud library
from wordcloud import WordCloud

# create a single string of space separated words
unique_string=(" ").join(words)

# create the word cloud and save to file
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
wordcloud.to_file("word_cloud.png")


