# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:59:31 2021

@author: theas
"""

#importing packages to read files and do manipulation
import pandas as pd
import matplotlib.pyplot as plt

telcom_data=pd.read_excel("D:/DataScience/Class/assignment working/h_clustering/Telco_customer_churn.xlsx")

#droping columns that are not relavent for our use
telcom_data_1=telcom_data.drop(['Customer ID','Count','Quarter','Referred a Friend'],axis=1)

from collections import Counter
Counter(telcom_data_1.Offer)

#creating dummies 
telcom_data_2=pd.get_dummies(telcom_data_1,drop_first=True)

#descritising data
telcom_data_2=telcom_data_2.astype(int)


#checking outliers
telcom_data_2.plot(kind="box",subplots=True,layout=(7,7),figsize=(20,15))

#    EDA

#cheking mean,mode,median
telcom_data_2.mean()
telcom_data_2.mode()
telcom_data_2.median()

#checking shewness
telcom_data_2.skew()

#checking kurtoisis
telcom_data_2.kurtosis()

#checking zero variance
telcom_data_2.var()
#nothing to remove

#checking standard deviations
telcom_data_2.std()

#normalizing scale 
def normalize(x):
    i=(x-x.min())/(x.max()-x.min())
    return i

norm_telcom=normalize(telcom_data_2)
#hirarchical clustering
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
#calculating distancs
link=linkage(norm_telcom,method="complete",metric="euclidean")

plt.figure(figsize=(30,15));plt.title("Clusters of Telco");plt.xlabel("index");plt.ylabel("Distances")
sch.dendrogram(link,p=5,truncate_mode="level",leaf_rotation=90,leaf_font_size=8)
plt.show()

from sklearn.cluster import AgglomerativeClustering
agg=AgglomerativeClustering(n_clusters=5 ,affinity="euclidean",linkage="complete").fit(norm_telcom)

#checking Clusters
agg.labels_

#adding cluster labels to main datafarame
telcom_data_1["Clusters"]=agg.labels_

#re-arrenging columns
telcom_data_1.columns=[ 'Clusters','Number of Referrals', 'Tenure in Months', 'Offer','Phone Service', 'Avg Monthly Long Distance Charges','Multiple Lines', 'Internet Service', 'Internet Type','Avg Monthly GB Download', 'Online Security', 'Online Backup','Device Protection Plan', 'Premium Tech Support', 'Streaming TV','Streaming Movies', 'Streaming Music', 'Unlimited Data','Contract', 'Paperless Billing', 'Payment Method','Monthly Charge', 'Total Charges', 'Total Refunds','Total Extra Data Charges', 'Total Long Distance Charges','Total Revenue']

#checking clusters
telcom_data_1.groupby(by="Clusters").mean()

#clustering with gower matrix

import gower
gow_matrix=gower.gower_matrix(telcom_data_1)

#distance calculation
link_1=linkage(gow_matrix)

plt.figure(figsize=(30,15));plt.title("Clusters of Telco");plt.xlabel("index");plt.ylabel("Distances")
sch.dendrogram(link_1,leaf_rotation=90,leaf_font_size=10,p=10,truncate_mode="level")
plt.show()

