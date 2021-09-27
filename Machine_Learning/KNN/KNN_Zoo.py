# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:15:59 2021

@author: theas
"""

#importing required pachages to perform data manipulation and handling

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

#importing data
zoo = pd.read_csv("D:/DataScience/Class/assignment working/KNN/Zoo.csv")

#checking desception
zoo.describe()

#checking requency of values

from collections import Counter

Counter(zoo["type"])

#dropping animal name as it wont help alorithrm learn in any way

X=zoo.drop(["animal name","type"],axis=1)
Y=zoo["type"]

#defining custom function

def norm(x):
    z=(x-x.min())/(x.max()-x.min())
    return z
X=norm(X)

#splitting data for training and testing
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=125)

#finding optimmal number of Neighbours

acc_score=[]
k_vals=[i for i in range(1,50,2)]

for k in k_vals:
    knn_1=KNeighborsClassifier(n_neighbors=k,n_jobs=-1,metric="euclidean")
    knn_1.fit(x_train,y_train)
    test_pred=np.mean(knn_1.predict(x_test)==y_test)
    train_pred=np.mean(knn_1.predict(x_train)==y_train)
    acc_score.append([train_pred,test_pred])


#visualizing accuracy with k values
plt.plot(k_vals,[i[0] for i in acc_score],"bo-")
plt.plot(k_vals,[i[1] for i in acc_score],"ro-")
plt.xlabel("K-values",fontsize=15)
plt.ylabel("Accuracy",fontsize=15)
plt.show()


#building KNN model with K=11

knn=KNeighborsClassifier(n_neighbors=13, n_jobs=-1,metric="euclidean")
knn.fit(x_train, y_train)

#predicting on test deta
test_pred=knn.predict(x_test) 
test_accuracy=accuracy_score(y_test, test_pred)
test_accuracy
pd.crosstab(y_test, test_pred,rownames=["Actual"],colnames=["predict"])

#checking accuracy on train data

train_predict=knn.predict(x_train)
train_accuracy=accuracy_score(y_train,train_predict)
train_accuracy
pd.crosstab(y_train, train_predict,rownames=["Actual"],colnames=["predict"])


