# -*- coding: utf-8 -*-
"""
Created on Wed May 19 21:23:25 2021

@author: theas
"""
#importing packages to read  and manipulate deta


#importing packages to read and manipulate deta
import pandas as pd
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
auto_dummies.agg(["mean","median","var","std","skew","kurt"])

def normalize(x):
    w=(x-x.min())/(x.max()-x.min())
    return w

norm_auto=normalize(auto_dummies)

# Kmean clusters formation
from sklearn.cluster import KMeans

#calculating inertia for different numbers of clusters
twss=[]
i=range(2,11)
for x in i:
    kmeans=KMeans(n_clusters=x)
    kmeans.fit(norm_auto)
    twss.append(kmeans.inertia_)

twss

#plotting scree plot
plt.plot(i,twss,"ro-");plt.xlabel("Number of Clusters");plt.ylabel("Total Winthin Sum of Square")

#at cluster 5 therenis maximum bend ,so choosing 5 clusters

#KMeans clustering

kmeans=KMeans(n_clusters=5)
kmeans.fit(norm_auto)

#checking labels and creating new column in original data frame

kmeans.labels_
auto["clusters"]=kmeans.labels_

#re arrenging clumns
auto.columns.values
auto=auto[['clusters','Customer', 'State', 'Customer Lifetime Value', 'Response','Coverage', 'Education', 'Effective To Date', 'EmploymentStatus','Gender', 'Income', 'Location Code', 'Marital Status','Monthly Premium Auto', 'Months Since Last Claim','Months Since Policy Inception', 'Number of Open Complaints','Number of Policies', 'Policy Type', 'Policy', 'Renew Offer Type','Sales Channel', 'Total Claim Amount', 'Vehicle Class','Vehicle Size']]

#grouping cluters
a=auto.groupby(by="clusters").mean()
