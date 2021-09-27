# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 19:50:28 2021

@author: theas
"""
import pandas as pd
'''
#The library requests is used to get the content from a web page
import requests

#to convert the content of webpage into proper format we are importing beutitul soup
from bs4 import BeautifulSoup as bs
note_10_reviews=[]
num=range(1,40)
for i in num:
    ip=[]
    link="https://www.amazon.in/Test-Exclusive_2020_1143-Multi-3GB-Storage/product-reviews/B089MSG56S/ref=cm_cr_getr_d_paging_btm_next_3?ie=UTF8&reviewerType=all_reviews&pageNumber="+str(i)
    page= requests.get(link)
    page
    soup = bs(page.content,"html.parser")
    
    reviews=soup.find_all('span',{"data-hook" : "review-body"})
    for x in range(0,len(reviews)):
        ip.append(reviews[x].get_text)
    note_10_reviews=note_10_reviews+ip
 
with open("note_10.txt","w",encoding="utf8") as ot:
    ot.write(str(note_10_reviews))
note_10_reviews=[]
'''

#ive already extracted and provided amazon review data under "note_10.txt"

with open ("D:/DataScience/Class/Assignments/NLP Text mining/note_10.txt","r",encoding="utf8") as am:
    note_10_reviews=am.read()

note_10=str(note_10_reviews)
import re
words_1=[]
words_1.append(re.sub('[^A-Za-z" "]+',' ',note_10).lower())
doc_list=note_10.split("\n")
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
custom_stop=["br","days","sd","mm","gr","com","xiaomi","mp","span","text","review","data","hook","class","bound","method","tag","eu","ssl","amazon","png","sl","con","mi","u","go","id","url","type","w","g","img","name","http","get '","content","get","base","html","input", "desktop","control","none", "source","block","http","span","image","phone"]
words_without_stop_1=[word for word in words_tok if word not in stop_w ]
words_without_stop=[word for word in words_without_stop_1 if word not in custom_stop]

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
