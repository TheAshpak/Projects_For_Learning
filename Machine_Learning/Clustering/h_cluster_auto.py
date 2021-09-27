# -*- coding: utf-8 -*-
"""
Created on Wed May 19 17:25:32 2021

@author: theas
"""

#importing packages to read and manipulate deta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

auto=pd.read_csv("D:/DataScience/Class/assignment working/h_clustering/AutoInsurance.csv")

#checking descreption
auto.describe()

#removing unwanted columns

auto_1=auto.drop(["Customer","State","Vehicle Size","Effective To Date","Location Code"],axis=1)

#creating dummy variables for categorical data
auto_1.isna().sum()

auto_dummies=pd.get_dummies(auto_1,drop_first=True).astype(int)

# EDA
eda=auto_dummies.agg(["mean","median","var","std","skew","kurt"])


def normalize(x):
    w=(x-x.min())/(x.max()-x.min())
    return w

norm_auto=normalize(auto_dummies)

#hirarchical clustering

#importing packages for hirachical clustering

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

#calculating distance
link=linkage(norm_auto,method="complete",metric="euclidean")

plt.figure(figsize=(30,15));plt.title("Dendrogram of AUTO INSURENCE");plt.xlabel("Index");plt.ylabel("Distance")
sch.dendrogram(link,p=5,truncate_mode="level")
plt.show()

#Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering

#calculating distances
agg= AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="complete").fit(norm_auto)

#addimg new column to dataframe of clusters
auto["Cluster"]=agg.labels_


#rearrenging columns
auto=auto.iloc[:,[24,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]

#getting clusters
auto.groupby(by="Cluster").mean()

#counting each clusters frequency
from collections import Counter
Counter(auto.Cluster)
