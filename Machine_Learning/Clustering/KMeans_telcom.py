# -*- coding: utf-8 -*-
"""
Created on Thu May 20 20:43:18 2021

@author: theas
"""
#importing packages to read files and do manipulation
import pandas as pd
import matplotlib.pyplot as plt

telcom_data=pd.read_excel("D:/DataScience/Class/assignment working/K-mean/Telco_customer_churn (1).xlsx")

#droping columns that are not relavent for our use
telcom_data_1=telcom_data.drop(['Customer ID','Count','Quarter','Referred a Friend'],axis=1)

from collections import Counter
Counter(telcom_data_1.Offer)

#creating dummies 
telcom_data_2=pd.get_dummies(telcom_data_1,drop_first=True)

#descritising data
telcom_data_2=telcom_data_2.astype(int)


#checking outliers
telcom_data_2.plot(kind="box",subplots=True,layout=(6,6),figsize=(30,15))

#    EDA

#cheking mean,mode,median

telcom_data_2.agg(["mean","median","var","std","skew","kurt"])



#normalizing scale 
def normalize(x):
    i=(x-x.min())/(x.max()-x.min())
    return i

norm_telcom=normalize(telcom_data_2)

#importing required libraries
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#ploting total withinn sum of quares
twss=[]
k=range(2,9)
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(norm_telcom)
    twss.append(kmeans.inertia_)
plt.plot(k, twss,"ro-");plt.xlabel("No_of_Clusters");plt.ylabel("total_within_Sum_of_Square")

#max bend is at 3 cluster, so chosing 3 clusters

kmeans=KMeans(n_clusters=3)
kmeans.fit(norm_telcom)
clusters_normal=telcom_data.copy()
#telcom_data["Clusters"]=kmeans.labels_
clusters_normal["Clusters"]=kmeans.labels_
clusters_normal.columns.values #it will return list of column names
clusters_normal=clusters_normal[['Clusters','Customer ID', 'Count', 'Quarter', 'Referred a Friend','Number of Referrals', 'Tenure in Months', 'Offer','Phone Service', 'Avg Monthly Long Distance Charges','Multiple Lines', 'Internet Service', 'Internet Type','Avg Monthly GB Download', 'Online Security', 'Online Backup','Device Protection Plan', 'Premium Tech Support', 'Streaming TV','Streaming Movies', 'Streaming Music', 'Unlimited Data','Contract', 'Paperless Billing', 'Payment Method','Monthly Charge', 'Total Charges', 'Total Refunds','Total Extra Data Charges', 'Total Long Distance Charges','Total Revenue']] #putting #clusters column 

#looking at mean of clusters
clusters_normal.groupby(by='Clusters').mean()

from gower import gower_matrix

dist=gower_matrix(telcom_data)
t=range(2,8)
TWSS=[]
for i in t:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(dist)
    TWSS.append(kmeans.inertia_)
plt.plot(t, TWSS,"ro-");plt.xlabel("No_of_Clusters");plt.ylabel("total_within_Sum_of_Square")
km=KMeans(n_clusters=3)
km.fit(dist)
b=pd.DataFrame(km.labels_)
cluster_gower=telcom_data.copy()
cluster_gower["cluster"]=b
cluster_gower.groupby(by="cluster").mean()
