# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 22:28:32 2021

@author: theas
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
dt= pd.read_csv("D:/DataScience/Class/Learning resorces/Recommondation system/anime.csv")

tfidf= TfidfVectorizer(stop_words="english",lowercase=True)

dt.genre.isna().sum()
dt=dt.dropna(subset=["genre"])
tfidf_matrix =tfidf.fit_transform(dt.genre)
#tfidf matrix create sparse matrix document wise
tfidf_matrix.shape
dt.shape

#caculating cosine similarity
from sklearn.metrics.pairwise import cosine_similarity ,linear_kernel
cos_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)
cos_sim_matrix_1 = cosine_similarity(tfidf_matrix,tfidf_matrix)

#linear_kernel and cosine_similarity matrix gives same output

movie_index = pd.Series(data=dt.index,index= dt["name"]).drop_duplicates(keep="first")

#defining custome function to calculate similarity based on movie

def get_recommondations(Movie, topN):
    #Movie="Money Train (1995)"
    #topN=10
    #getting movie id from movie_index df
    movie_id = movie_index[Movie]
    
    #getting score of this movie id
    scores = list(enumerate(cos_sim_matrix[movie_id]))
    
    #getting topN
    top= sorted(scores, key= lambda x : x[1], reverse=True)
    #excliuding first as it the same name of provided movie
    top_N = top[0:topN+1]    
    movie_idx = [i[0] for i in top_N]
    movie_scores = [i[1] for i in top_N]
    
    #creating new dataframe to store scores and name of this movies
    recom = pd.DataFrame(columns=["Name","Score"])
    recom.Name= dt.loc[movie_idx ,"name" ]
    recom.Score = movie_scores
    recom.reset_index(inplace=True)
    print(recom)      

get_recommondations("Avengers: Age of Ultron (2015)", 10)
