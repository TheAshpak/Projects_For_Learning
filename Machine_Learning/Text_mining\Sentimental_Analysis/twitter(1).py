# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:40:34 2021

@author: theas
"""
#way to extract tweets from twitter
'''
#pip install tweepy

import tweepy as tw

#extracting tweets of Current president of United State 

#authenticating user
consumer_key=  " "
consumer_secret= ' '
access_token= ' '
access_token_secret= ' '
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

#initializing user name to extract tweets
userID="JoeBiden"
tweets = api.user_timeline(screen_name=userID, 
                           # 200 is the maximum allowed count
                           count=200,
                           include_rts = False,
                           # Necessary to keep full_text 
                           # otherwise only the first 140 words are extracted
                           tweet_mode = 'extended'
                           )
from credentials import *
#pip install credentials

#now extracting maximum number of tweets as possible
all_tweets = []
all_tweets.extend(tweets)
oldest_id = tweets[-1].id
while True:
    tweets = api.user_timeline(screen_name=userID, 
                           # 200 is the maximum allowed count
                           count=200,
                           include_rts = False,
                           max_id = oldest_id - 1,
                           # Necessary to keep full_text 
                           # otherwise only the first 140 words are extracted
                           tweet_mode = 'extended'
                           )
    if len(tweets) == 0:
        break
    oldest_id = tweets[-1].id
    all_tweets.extend(tweets)
    print('N of tweets downloaded till now {}'.format(len(all_tweets)))

#checking working directory
import os
os.getcwd()

#saving tweets to Document variable
documents=[]
for info in all_tweets:
     documents.append(info.full_text)
df=pd.DataFrame(documents,columns=["Tweets"])
df.to_csv("tweets_1.csv",index=False)
'''
#i have already extracted and saved tweets in tweets.csv and have provided it with  codes


#importing pandas to read file and do manipulation
import pandas as pd
import numpy as np
tweets=pd.read_csv("D:/DataScience/Class/Assignments/NLP Text mining/tweets.csv")

#importing nltk library for text mining 
import nltk

#importing re package to clean corpus
import re

# cleaning corpus
words_1=[]
for x in tweets.Tweets:
    words_1.append(re.sub('[^A-Za-z" "]+',' ',x).lower())
words_text=" ".join(words_1)

words_string=words_text.split(" ")

words_string=[word for word in words_string if word != ""]
#removing white spaces
from nltk import WhitespaceTokenizer
space=WhitespaceTokenizer()
words_tok=space.tokenize(words_text)

#removing stop words 
from nltk.corpus import stopwords
stop_w=stopwords.words("English")
stop_w.extend(["co","u","https","xnnfn","hfa","ewdq","eoxt","dq"])
words_without_stop=[word for word in words_tok if word not in stop_w ]

#importing lemmanizer to lemmanize words
from nltk import WordNetLemmatizer
from nltk import word_tokenize
lem=WordNetLemmatizer()

#getting root words
words=" ".join([lem.lemmatize(word) for word in words_without_stop ])

#importing words clouse and printing cloude
import matplotlib.pyplot as plt

#pip install WordCloud
from wordcloud import WordCloud

from collections import Counter

words_tok=word_tokenize(words)
count=Counter(words_tok)
freq=sorted(count.items() ,key = lambda x:x[1])
top=list(reversed(freq))

word_dict= dict(top)
WC_height = 1000
WC_width = 1500
WC_max_words = 350
wordCloud = WordCloud(max_words = WC_max_words,height= WC_height, width=WC_width ,stopwords=stop_w) 
wordCloud.generate_from_frequencies(word_dict)
plt.figure(figsize=(30,30))
plt.title("Word cloud of US President Bieden",fontsize=60)
plt.imshow(wordCloud,interpolation="bilinear")
plt.show()


#doing sentimental analysis

#importing text blob to calculate sentimental ppolarity

from textblob import TextBlob
sent=TextBlob(words).sentiment.polarity
sent
#we can see overall its positive tweets 
words_2=[lem.lemmatize(x) for x in words_1]
df=pd.DataFrame(words_2,columns=["Review"])
df.reset_index(inplace=True)
polarity=[]
df.shape

#calculating sentiments
for x in range(0,df.shape[0]):
    score=TextBlob(df.iloc[x][1])   #getting each tweet and passing it into TextBlob to further calculate polarity
    score_1 = score.sentiment[0]
    polarity.append(score_1)
df["score"]=polarity
#representing analysis in dataframe
positive=len(df[df.score>0])
negative=len(df[df.score<0])
neutral=len(df[df.score==0])
p=np.array([positive,negative,neutral])
mylabels=["positive","negative","neutral"]

plt.figure(figsize=(10,10))
plt.title("Sentiments used by US Pesiden in recent tweets")
plt.pie(p,labels=["positive","negative","neutral"])
plt.show()


