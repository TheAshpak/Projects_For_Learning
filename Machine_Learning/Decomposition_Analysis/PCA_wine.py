# -*- coding: utf-8 -*-
"""
Created on Fri May 21 22:15:04 2021

@author: theas
"""
#importing library to read and manipulate deta

import pandas as pd 
import numpy as np

#loading data into panda to manipulate

wine_data = pd.read_csv("D:/DataScience/Class/assignment working/PCA/wine.csv")

#checking descriptions

wine_data.describe()

#cheling duplicates
wine_data.duplicated().sum()

#checking zero variance
wine_data.var()  #Three columns have near zero variance so they should be removed, as they wont help in model leanring 

wine_data_zvar=wine_data.drop(["Nonflavanoids","Ash","Hue"],axis=1)

#cheking null values
wine_data.isna().sum()

#outliers analysis and treatments

wine_data.plot(kind="box",subplots=True,layout=(4,4),figsize=(15,8))
wine_data_zvar.columns.values
cols=['Malic', 'Alcalinity', 'Magnesium','Proanthocyanins','Color']
win_data_n_out=pd.DataFrame()

#winsorizing data
from scipy.stats.mstats import winsorize

for col in cols:
    win_data_n_out[col]=winsorize(wine_data[col],limits=[0.05,0.05])

#checking whether winsorization wored fine
win_data_n_out.plot(kind="box",subplots=True,layout=(4,4),figsize=(15,8))

#concat remaining cols into winsorized data
wine_data_n_out=pd.concat([wine_data_zvar.loc[:,['Type', 'Alcohol', 'Phenols','Flavanoids', 'Dilution', 'Proline']],win_data_n_out],axis=1)

wine_data_final=pd.get_dummies(wine_data_n_out,columns=["Type"],drop_first=True)

#scaling data
def norm(x):
    z=(x-x.min())/(x.max()-x.min())
    return z
norm_wine=norm(wine_data_final)

#PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca_wine=PCA(n_components=12)
pca_wine_values = pca_wine.fit_transform(norm_wine)
pca_var_wine=pca_wine.explained_variance_ratio_

pca_wine.components_

# Cumulative variance 
var_1=np.cumsum(np.round(pca_var_wine,decimals=4)*100)
var_1

# Variance plot for PCA components obtained 
plt.plot(var_1, color = "red")

# PCA scores
pca_wine_values

#retriving original columns name
pca_data_wine = pd.DataFrame(pca_wine_values)
pca_data_wine.columns = norm_wine.columns

#cheking coliniarity between first two components

plt.scatter(x=pca_data_wine.Alcohol,y=pca_data_wine.Phenols)

#performing Hirarchical clustering

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

#calculating distance
link=linkage(pca_data_wine.iloc[:,0:3],method="complete",metric="euclidean")

#ploting dendrogram
plt.figure(figsize=(15,8));plt.ylabel("Distance");plt.xlabel("Index");plt.title("Wine_Dendrogram")
sch.dendrogram(link,p=10,truncate_mode="level",leaf_font_size=8,leaf_rotation=90)
plt.show()

#now forming clusters with Agglomerative clustering

from sklearn.cluster import AgglomerativeClustering
agg=AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="complete").fit(pca_data_wine.iloc[:,0:3])
h_clusters=wine_data.copy()
h_clusters["h_clust"]=agg.labels_
h_clusters.columns
h_clusters=h_clusters[['h_clust','Type', 'Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols','Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 'Hue','Dilution', 'Proline']]

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
    kmeans.fit(pca_data_wine.iloc[:,0:3])
    twss.append(kmeans.inertia_)
#plotting Elbow plot
plt.plot(x,twss);plt.title("Elbow plot");plt.xlabel("Clusters");plt.ylabel("TWSS")

#maximum bend at 3rd cluster so choosing 3 clusters

#forming clusters with KMeans
kmeans=KMeans(n_clusters=3)
kmeans.fit(pca_data_wine.iloc[:,0:3])
kmeans_cluster=wine_data.copy()
kmeans_cluster["k_clust"]=kmeans.labels_
#shiffling columns
kmeans_cluster.columns
kmeans_cluster=kmeans_cluster[['k_clust','Type', 'Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols','Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 'Hue','Dilution', 'Proline']]
#plotting clusters
kmeans_cluster.groupby(by="k_clust").mean()

#Kmeans clustering without PCA
k=range(2,8)
TWSS=[]
for i in k:
    kmean=KMeans(n_clusters=i)
    kmean.fit(norm_wine)
    TWSS.append(kmean.inertia_)
plt.plot(k,TWSS);plt.title("Elbow plot");plt.xlabel("Clusters");plt.ylabel("TWSS")

#maximum bend at 3 clusters, so choosing 3 clusters
kmeans_o=KMeans(n_clusters=3)
kmeans_o.fit(norm_wine)
kmeans_o.labels_

#KMeans giving same number of clusters for data with pca and data without pca
