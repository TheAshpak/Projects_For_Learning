# -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:07:05 2021

@author: theas
"""
#importing require packages for manipulation of data
import pandas as pd
import numpy as np

#loading package to the pandas dataframe
movie=pd.read_csv("D:\DataScience\Class\Assignments\Recommondation system/Entertainment.csv")
movie.shape #checking shape of data

#looking at data 
movie.head()

#impporting TfidVectorizer to remove all stop words from data set
from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 

# replacing the NaN values in overview column with empty string
movie["Category"].isnull().sum()
#no null

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(movie.Category)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #51,34

# with the above matrix we need to find the similarity score
# There are several metrics for this such as the euclidean, 
# the Pearson and the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity between 2 movies 
# Cosine similarity - metric is independent of magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)
#to calculate cosine values importing linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of movie name to index number 
movie_index = pd.Series(movie.index, index = movie['Titles']).drop_duplicates(keep="first")

#defining custom function to calculate similar movies

def get_recommendations(Name, topN):    
    #Name="Grumpier Old Men (1995)"
    #topN=10
    # Getting the movie index using its title 
    movie_id = movie_index[Name]
    
    # Getting the pair wise similarity score for all the movie's with that 
    # movie
    cosine_scores = list(enumerate(cosine_sim_matrix[movie_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    movie_idx  =  [i[0] for i in cosine_scores_N]
    movie_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    movie_similar_show = pd.DataFrame(columns=["name", "Score"])
    movie_similar_show["name"] = movie.loc[movie_idx, "Titles"] #Title is from movie ,basically we are giGrumpier Old Men (1995)ing two index to .loc
    movie_similar_show["Score"] = movie_scores
    movie_similar_show.reset_index(inplace = True)  
    # movie_similar_show.drop(["index"], axis=1, inplace=True)
    print (movie_similar_show)
    # return (movie_similar_show)

    
# Enter your movie and number of movie's to be recommended 
get_recommendations("Balto (1995)", topN = 10)
movie_index["Balto (1995)"]
