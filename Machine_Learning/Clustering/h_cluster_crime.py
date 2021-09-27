# -*- coding: utf-8 -*-
"""
Created on Tue May 18 22:52:00 2021

@author: theas
"""
#importing packages to read amd manipulate data
import pandas as pd
import numpy as np
crime_data=pd.read_csv("D:/DataScience/Class/assignment working/h_clustering/crime_data.csv")

#checking EDA

crime_data.describe()
crime_data.columns.values
#removing categorical column
crime=crime_data.drop("Unnamed: 0",axis=1)

#type casting
crime=crime.astype("int")

#checking missing values
crime.isna().sum()

#checking outliers
crime.plot(kind="box",subplots=True,layout=(4,4),figsize=(15,8))

#outlier treatment
from scipy.stats.mstats import winsorize
crime["Rape"]=winsorize(crime["Rape"],limits=(0.01,0.04))

crime["Rape"].plot(kind="box")

#zero variance
crime.var()

#normalizing data
def norm(x):
    z=(x-x.min())/(x.max()-x.min())
    return z

crime_norm=norm(crime)

#hirarchical clustering

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
#calculating distances
link=linkage(crime_norm,method="complete",metric="euclidean")

#plotting dendrogram
plt.figure(figsize=(15,8));plt.title("Dendrogram of Crime Data");plt.xlabel("Index");plt.ylabel("Distance")
sch.dendrogram(link)
plt.show()

#forming clusters
from sklearn.cluster import AgglomerativeClustering
clusters=AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="complete").fit(crime_norm)
crime_data["cluster"]=clusters.labels_
crime_data=crime_data.iloc[:,[5,0,1,2,3,4]]
crime_data.groupby(by="cluster").mean()
