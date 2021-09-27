# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:56:42 2021

@author: theas
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
airline=pd.read_excel("D:/DataScience/Class/assignment working/K-mean/EastWestAirlines (1).xlsx",1)
#lokking at data types
airline.head()

#checking EDA
airline.describe()

#checkinh null values
airline.isna().sum()

#checking data types
airline.dtypes

#checking Duplicates
airline.duplicated().sum()

#Checking outliers
airline.plot(kind="box",subplots=True,layout=(4,4),figsize=(30,15))

#outliers treatment

q1=airline["Balance"].quantile(0.25)
q3=airline["Balance"].quantile(0.75)
H_limit=q3+1.5*(q3-q1)
win_quant=airline.Balance.quantile(0.93)
airline['Balance']=np.where(airline["Balance"]>H_limit,win_quant,airline["Balance"])
airline["Balance"].plot(kind="box")


airline["Qual_miles"].describe()
#droping ID
airline_1=airline.drop(["ID#"],axis=1)

airline_1.var()

#defining function to normalizing data

def normalize(x):
    q=(x-x.min())/(x.max()-x.min())
    return q
#normalizing data
norm_airline=normalize(airline)

#Starting clustering
from sklearn.cluster import KMeans
twss=[]
s=range(2,9)
for x in s:
    kmeans=KMeans(n_clusters=x)
    kmeans.fit(norm_airline)
    twss.append(kmeans.inertia_)

#plotting scree plot
plt.plot(s,twss,"ro-");plt.xlabel("Numbers Of Clusters");plt.ylabel("Total withiness")

#as from scree plot at cluster =5 there is maximun bend, selecting 5 clusters

kmeans=KMeans(n_clusters=5)
kmeans.fit(norm_airline)


airline_1["Clusters"]=kmeans.labels_
airline_1.head()
airline_1=airline_1[['Clusters','Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles','Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo','Flight_trans_12', 'Days_since_enroll', 'Award?']]

#printing clusters
airline_1.groupby(by="Clusters").mean()
