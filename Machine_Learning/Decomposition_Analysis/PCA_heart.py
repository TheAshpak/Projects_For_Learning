# -*- coding: utf-8 -*-
"""
Created on Sun May 23 00:09:13 2021

@author: theas
"""

#importing library to read and manipulate deta

import pandas as pd 
import numpy as np

#loading data into panda to manipulate

heart_data = pd.read_csv("D:/DataScience/Class/assignment working/PCA/heart disease.csv")

#checking descriptions

heart_data.describe()

#cheling duplicates
heart_data.duplicated().sum()

#dropping duplicates
heart_data=heart_data.drop_duplicates(keep="first")
#checking zero variance
heart_data.var()


#cheking null values
heart_data.isna().sum()

#outliers analysis and treatments

heart_data.plot(kind="box",subplots=True,layout=(4,4),figsize=(15,8))

#removing outliers
from scipy.stats.mstats import winsorize
heart_winsorize=pd.DataFrame()
heart_data.columns
for i in heart_data:
    heart_winsorize[i]=winsorize(heart_data[i],limits=[0.05,0.05])

heart_winsorize["ca"]=winsorize(heart_data["ca"],limits=[0.05,0.1])
heart_winsorize.plot(kind="box",subplots=True,layout=(4,4),figsize=(15,8))

#normalizing data
def norm(x):
    z=(x-x.min())/(x.max()-x.min())
    return z
norm_heart=norm(heart_winsorize)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca_heart=PCA(n_components=14)
pca_heart_values=pca_heart.fit_transform(norm_heart)

pca_var_heart=pca_heart.explained_variance_ratio_
pca_heart.components_
# Cumulative variance 
var_1=np.cumsum(np.round(pca_var_heart,decimals=4)*100)
var_1

# Variance plot for PCA components obtained 
plt.plot(var_1, color = "red")

# PCA scores
pca_heart_values

#retriving original columns name
pca_data_heart = pd.DataFrame(pca_heart_values)
pca_data_heart.columns = norm_heart.columns


#cheking coliniarity between first two components

plt.scatter(x=pca_data_heart.age,y=pca_data_heart.sex)

#performing Hirarchical clustering

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

#calculating distance
link=linkage(pca_data_heart,method="complete",metric="euclidean")

#ploting dendrogram
plt.figure(figsize=(15,8));plt.ylabel("Distance");plt.xlabel("Index");plt.title("Heart_Dendrogram")
sch.dendrogram(link,p=10,truncate_mode="level",leaf_font_size=8,leaf_rotation=90)
plt.show()

#now forming clusters with Agglomerative clustering

from sklearn.cluster import AgglomerativeClustering
agg=AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="complete").fit(pca_data_heart)
h_clusters=heart_data.copy()
h_clusters["h_clust"]=agg.labels_
h_clusters.columns
h_clusters=h_clusters[['h_clust','age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']]

#printing clusters
h_clusters.groupby(by="h_clust").mean()


#clustering with KMeans

#importing KMeans

from sklearn.cluster import KMeans

#finding optimal number of clusters with TWSS
twss=[]
x=range(2,8)
for i in x:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(pca_data_heart)
    twss.append(kmeans.inertia_)
#plotting Elbow plot
plt.plot(x,twss);plt.title("Elbow plot");plt.xlabel("Clusters");plt.ylabel("TWSS")

#maximum bend at 3rd cluster so choosing 3 clusters

#forming clusters with KMeans
kmeans=KMeans(n_clusters=4)
kmeans.fit(pca_data_heart)
kmeans_cluster=heart_data.copy()
kmeans_cluster["k_clust"]=kmeans.labels_
#shiffling columns
kmeans_cluster.columns
kmeans_cluster=kmeans_cluster[['k_clust','age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']]
#plotting clusters
kmeans_cluster.groupby(by="k_clust").mean()

#Kmeans clustering without PCA
k=range(2,8)
TWSS=[]
for i in k:
    kmean=KMeans(n_clusters=i)
    kmean.fit(norm_heart)
    TWSS.append(kmean.inertia_)
plt.plot(k,TWSS);plt.title("Elbow plot");plt.xlabel("Clusters");plt.ylabel("TWSS")

#maximum bend at 4 clusters, so choosing 4 clusters
kmeans_o=KMeans(n_clusters=4)
kmeans_o.fit(norm_heart)
kmeans_o.labels_

#KMeans giving same number of clusters for data with pca and data without pca