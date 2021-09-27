# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 20:14:26 2021

@author: theas
"""
#importing required libraries
import pandas as pd
import numpy as np

#importing data set
car=pd.read_csv("D:/DataScience/Class/assignment working/Naive Bayes/NB_Car_Ad.csv")

#checking data
car.head()

#checking null value
car.isna().sum()

#droping unwanted columns
car.drop("User ID",inplace=True,axis=1)

#creating dummies of categorical columns
car=pd.get_dummies(car,columns=["Gender"])

#chreating target
target=car['Purchased']

#creating inputs
inputs=car.drop("Purchased",axis=1)

#importing library to splite data set
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2,random_state=125)

#navie bayes for numeric data
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
train_nb=nb.fit(x_train, y_train)
x_train_pred=train_nb.predict(x_test)
test_accuracy=np.mean(x_train_pred==y_test)
test_accuracy

pd.crosstab(x_train_pred,y_test)

train_predict=train_nb.predict(x_train)
train_accuracy=np.mean(y_train==train_predict)
train_accuracy
pd.crosstab(y_train, train_predict)

#applying smoothning to eliminate false positve values

nb=GaussianNB(var_smoothing=1e-8)
train_nb=nb.fit(x_train, y_train)
x_train_pred=train_nb.predict(x_test)
test_accuracy=np.mean(x_train_pred==y_test)
test_accuracy

pd.crosstab(x_train_pred,y_test)

train_predict=train_nb.predict(x_train)
train_accuracy=np.mean(y_train==train_predict)
train_accuracy
pd.crosstab(y_train, train_predict)
