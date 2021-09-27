# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:12:41 2021

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


crime.plot(kind="box",subplots=True,layout=(4,4),figsize=(15,8))

#zero variance
crime.var()

#normalizing data
def norm(x):
    z=(x-x.min())/(x.max()-x.min())
    return z

crime_norm=norm(crime)

#plotting scree plot
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
twss=[]
x=range(2,9)
for i in x:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(crime_norm)
    twss.append(kmeans.inertia_)
plt.plot(x,twss,"ro-");plt.xlabel("No_of_Clusters");plt.ylabel("total_within_Sum_Of_Squares")

#making clusters
kmeans=KMeans(n_clusters=4)
kmeans.fit(crime_norm)
crime_data["Clusters"]=kmeans.labels_
crime_data.columns.values
crime_data=crime_data[['Clusters','Unnamed: 0', 'Murder', 'Assault', 'UrbanPop', 'Rape']]

crime_data.groupby(by="Clusters").mean()



