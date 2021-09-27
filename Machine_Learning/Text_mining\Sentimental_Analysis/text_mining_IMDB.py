# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:04:42 2021

@author: theas
"""
#importing required libraries
import requests
from bs4 import BeautifulSoup as bs
import re
import pandas as pd
import numpy as np

'''
url="https://www.imdb.com/title/tt0903747/reviews?ref_=tt_sa_3"

ip=[]
page= requests.get(url)
page
soup = bs(page.content,"html.parser")
reviews=soup.find_all("div",attrs={"text show-more__control"})

with open("breaking_bad.txt","w",encoding="utf8") as ot:
    ot.write(str(reviews))
'''
#ive extracted and provided reviews in  "breaking_bad.txt"

breaking=[]
with open("D:/DataScience/Class/assignment working/NLP Text mining/breaking_bad.txt",encoding="utf8") as o:
    breaking=o.read()


import re
words_1=[]
words_1.append(re.sub('[^A-Za-z" "]+',' ',breaking).lower())
doc_list=breaking.split("\n")
word_2=str(words_1)

word_2=re.sub('[^A-Za-z" "]+',' ',word_2)
word_2=re.sub('["" "]+',' ',word_2)

#importing package to toknaize white space
from nltk import WhitespaceTokenizer
space=WhitespaceTokenizer()

#removing white spaces and toaknizing words 
words_tok=space.tokenize(word_2)

#removing stop words and importing stop words from nltk corpus package
from nltk.corpus import stopwords
stop_w=stopwords.words("English")
stop_w.extend(["br","days","div","class","text","forward","series","control","show","season","episode","one"])
words_without_stop=[word for word in words_tok if word not in stop_w ]

#importing lemmenizer to llemenize words
from nltk import WordNetLemmatizer
from nltk import word_tokenize

#extracting root words
lem=WordNetLemmatizer()
words=" ".join([lem.lemmatize(word) for word in words_without_stop ])

#plotting word cloude
import matplotlib.pyplot as plt
#pip install WordCloud
from wordcloud import WordCloud


#unigram word cloude
plt.figure(figsize=(30,30))
wordcloud_ip = WordCloud(
    background_color="black",
    width = 1800,
    height = 1400
    ).generate(words)
plt.imshow(wordcloud_ip)

#bi-gram wordcloude
import nltk
from nltk import word_tokenize
token=word_tokenize(words)

#creting bigram
bigram_list = list(nltk.bigrams(token))

print(bigram_list) 

dictionary2 = [" ".join(bigram) for bigram in bigram_list]
print(dictionary2)

#using count vectoriser to view the frequency of bigrams

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words= vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

#calculating rowsum
sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, columnname]) for word, columnname in vectorizer.vocabulary_.items()]
word_freq = sorted(words_freq,key = lambda x: x[1], reverse=True)

#word cloude
word_dict= dict(word_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud(max_words = WC_max_words,height= WC_height, width=WC_width ,stopwords=stop_w) 
wordCloud.generate_from_frequencies(word_dict)
plt.figure(figsize=(30,30))
plt.title("Bigram word cloud")
plt.imshow(wordCloud,interpolation="bilinear")
plt.show()


#calculating sentiments

from textblob import TextBlob
sent=TextBlob(words).sentiment.polarity
sent
#we can see overall its positive review 