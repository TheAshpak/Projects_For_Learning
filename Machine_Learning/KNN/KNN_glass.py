# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:43:56 2021

@author: theas
"""

#importing required packages for data handling and manipulation

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#loading deta set
glass=pd.read_csv("D:/DataScience/Class/assignment working/KNN/glass.csv")

#checking description
glass.describe()


#checking frequency of values
from collections import Counter
Counter(glass["Type"])

#splitting data frame into input and output
X=glass.drop("Type",axis=1)
Y=glass["Type"]

##Normalizing data 

#defining custom function

def norm(x):
    z=(x-x.min())/(x.max()-x.min())
    return z
norm_input=norm(X)

#splitting data into training and testing
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

k_vals=[i for i in range(1,50,2)]

#empty list to hold cv score
test_k_acc_score = []
train_k_acc_score=[]
#cross validating K value
for k in k_vals:
    knn = KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
    knn.fit(x_train,y_train)
    test_pred=np.mean(knn.predict(x_test)==y_test)
    train_pred=np.mean(knn.predict(x_train)==y_train)
    test_k_acc_score.append(test_pred)
    train_k_acc_score.append(train_pred)


#plotting test and train accuracy
plt.plot(k_vals,[i for i in test_k_acc_score],"ro-")
plt.plot(k_vals,[i for i in train_k_acc_score],"go-")
plt.xlabel("K-values",fontsize=15)
plt.ylabel("Accuracy",fontsize=15)
plt.show()

#fitting KNN
knn=KNeighborsClassifier(n_neighbors=17,p=2,n_jobs=-1,metric="euclidean")

#training on train data
knn.fit(x_train,y_train)

#testing on test data
test_predict=knn.predict(x_test)
test_accuracy=accuracy_score(y_test, test_predict)
test_accuracy
pd.crosstab(y_test, test_predict,rownames=["Actual"],colnames=["predict"])
print(f1_score(y_test, test_predict,average='micro'))
#checking accuracy on train data

train_predict=knn.predict(x_train)
train_accuracy=accuracy_score(y_train,train_predict)
train_accuracy
pd.crosstab(y_train, train_predict,rownames=["Actual"],colnames=["predict"])
