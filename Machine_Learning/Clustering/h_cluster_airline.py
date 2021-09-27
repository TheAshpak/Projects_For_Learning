# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:44:38 2021

@author: theas
"""
#loading packages to read and do manipulation on given data
import pandas as pd
import numpy as np

#reading excel file
excel = pd.read_excel("D:/DataScience/Class/assignment working/h_clustering/EastWestAirlines.xlsx",1)

#lokking at data types
excel.head()

#checking EDA
excel.describe()

#checkinh null values
excel.isna().sum()

#checking data types
excel.dtypes

#checking Duplicates
excel.duplicated().sum()

#Checking outliers
excel.plot(kind="box",subplots=True,layout=(4,4),figsize=(30,15))

#outliers treatment

q1=excel["Balance"].quantile(0.25)
q3=excel["Balance"].quantile(0.75)
H_limit=q3+1.5*(q3-q1)
win_quant=excel.Balance.quantile(0.93)
excel['Balance']=np.where(excel["Balance"]>H_limit,win_quant,excel["Balance"])
excel["Balance"].plot(kind="box")


excel["Qual_miles"].describe()
#droping ID
excel_1=excel.drop(["ID#"],axis=1)

excel_1.var()

#cc2_miles and cc3_miles have near zero variance so it wont help in model lerning hence removing them
excel_1=excel_1.drop(["cc2_miles","cc3_miles"],axis=1)

#normalizing data

def norm(x):
    z=(x-x.min())/(x.max()-x.min())
    return z
norm_data=norm(excel_1)

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

link=linkage(norm_data,method="complete",metric="euclidean")
plt.figure(figsize=(30,15));plt.title("Dendrogram of EastWestAirline");plt.xlabel("Index");plt.ylabel("Distance")
sch.dendrogram(link,p=10,truncate_mode="level")
plt.show()

# Now applying AgglomerativeClustering 
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(norm_data) 
h_complete.labels_


excel_1["clust"] = h_complete.labels_ # creating a new column and assigning it to new column 


excel_1.head()

# Aggregate mean of each cluster
excel_1.groupby("clust").mean()
