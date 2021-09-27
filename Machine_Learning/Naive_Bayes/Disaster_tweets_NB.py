# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 22:34:31 2021

@author: theas
"""

#importing required library to import deta set and manipulate

import pandas as pd
import numpy as np

#loading file into spider

tweets=pd.read_csv("D:/DataScience/Class/assignment working/Naive Bayes/Disaster_tweets_NB.csv")

#checking data
tweets.head()

#checking descriptions
tweets.describe()

#checking missing values
tweets.isna().sum()

#location has maximum missing values so its better to drop it
tweets_1=tweets.drop(["location","id"],axis=1)

#droping rows with NA (0.8% data)
tweets_1.dropna(how="any",inplace=True)

#cleaning tweets
import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

#initilizing lemmatizer

lemmeatizer=WordNetLemmatizer()
lemmeatizer.lemmatize("programs")
#creating custom function to perfor cleaning and stemming

#importing stemmer
from nltk.stem import PorterStemmer
stem=PorterStemmer()

#writing custom function
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            word=lemmeatizer.lemmatize(word)
            w.append(word)
    return (" ".join(w))

#cleaning and lemmatizing data
tweets_1.text=tweets_1.text.apply(cleaning_text)
tweets_1.keyword=tweets_1.keyword.apply(cleaning_text)

#custom tokenizing data with custom function   
'''
def tokn(i):
    i=word_tokenize(i)
    return i
tweets_1.text=tweets_1.text.apply(tokn)
tweets_1.keyword=tweets_1.keyword.apply(tokn)
'''

from sklearn.model_selection import train_test_split

train,test=train_test_split(tweets_1,test_size=0.2,random_state=43)


# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email texts into word count matrix format - Bag of Words
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
#analyzer = split_into_words
tweets_bow = CountVectorizer(analyzer = split_into_words).fit(tweets_1.text)

# Defining BOW for all messages
input_tweets_matrix = tweets_bow.transform(tweets_1.text)
input_tweets_matrix.shape
# For training messages
train_matrix = tweets_bow.transform(train.text)
#train_matrix.shape

# For testing messages
test_matrix = tweets_bow.transform(test.text)
#test_matrix.shape
# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(input_tweets_matrix)
# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_matrix)
#train_tfidf.shape # (row, column)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_matrix)
#test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB
naive_bayes=MB()
fit_data=naive_bayes.fit(train_tfidf,train.target)
test_predict=fit_data.predict(test_tfidf)

#checking test accuracy
from sklearn.metrics import accuracy_score
test_accuracy=accuracy_score(test_predict,test.target)
test_accuracy

#checking cross tab

pd.crosstab(test_predict,test.target)

# Training Data accuracy
train_predict= fit_data.predict(train_tfidf)
train_accuracy=accuracy_score(train_predict,train.target)
train_accuracy

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.



naive_bayes_lap = MB(alpha = 25)
naive_bayes_lap.fit(train_tfidf,train.target)

# Evaluation on Test Data after applying laplace
test_pred_lap = naive_bayes_lap.predict(test_tfidf)
test_accuracy_lap = accuracy_score(test_pred_lap,test.target) 
test_accuracy_lap

#checking false positive and false negative
pd.crosstab(test_pred_lap, test.target)


# Training Data accuracy
train_pred_lap = naive_bayes_lap.predict(train_tfidf)
train_accuracy_lap = accuracy_score(train_pred_lap,train.target)
train_accuracy_lap

#checking false positive and false negative
pd.crosstab(train_pred_lap,train.target)

