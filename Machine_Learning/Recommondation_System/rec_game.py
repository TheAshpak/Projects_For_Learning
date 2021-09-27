# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:18:42 2021

@author: theas
"""
#importing libraries required for data processing
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")
%matplotlib inline

#importing dataset with pandas
game=pd.read_csv("D:/DataScience/Class/assignment working/Recommondation system/game.csv")

#checking top five rows
game.head()
#calculating mean of ratings
games_mean=game.iloc[:,1:].groupby(by="game")["rating"].mean()

#calculating number of ratings given for each movies
rating_count=game.groupby(by="game")["rating"].count()

#creating new data frame
game_mean_count=pd.DataFrame(games_mean)
game_mean_count["rating_count"]=rating_count

#plotting graphs with numbers of ratings and ratings
plt.figure(figsize=(6,4))
game_mean_count["rating_count"].hist(bins=50)

plt.figure(figsize=(6,4))
game_mean_count["rating"].hist(bins=50)

#plotting joint plot 
plt.figure(figsize=(6,4))
sns.jointplot(x="rating",y="rating_count",data=game_mean_count,alpha=0.4)

#creating pivote table to calculate distances
game_mat = game.pivot_table(index="userId",columns='game',values='rating')

#calculating nearest distances for "A way out"
ratings=game_mat["A Way Out"]

#finding neighbours
similar_games=game_mat.corrwith(ratings)
corr=pd.DataFrame(similar_games,columns=["correlation"])

corr.dropna(inplace=True)
print(corr)


#we dont have much number of ratings for each movies to find any correlation between movies
