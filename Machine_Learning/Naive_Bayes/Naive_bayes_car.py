# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 20:30:21 2021

@author: theas
"""

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

#Checking distribution of data
import matplotlib.pyplot as plt

plt.hist(car.EstimatedSalary, bins=50)
plt.show()

#normalizing data with custome function
#defining custom function
def norm(x):
    z=(x-x.min())/(x.max()-x.min())
    return z
norm_data=norm(inputs)

#importing library to split data set
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2,random_state=117)

#navie bayes for numeric data
from sklearn.naive_bayes import BernoulliNB

#initializing bernouliiNB
nb=BernoulliNB()

#fitting data
train_nb=nb.fit(x_train, y_train)

#testing on test data
x_train_pred=train_nb.predict(x_test)

#checking accuracy
test_accuracy=np.mean(x_train_pred==y_test)
test_accuracy

#checking cross tab
pd.crosstab(x_train_pred,y_test)

#checking accuracy on train deta how much model learn
train_predict=train_nb.predict(x_train)
train_accuracy=np.mean(y_train==train_predict)

#checking accuracy
train_accuracy
pd.crosstab(y_train, train_predict)

#applying smoothning to eliminate false positve values

#initializing bernoullies

nb_l=BernoulliNB(alpha=1)

#training data
train_nb=nb_l.fit(x_train, y_train)
x_train_pred_l=train_nb.predict(x_test)

#cheking accuracy
test_accuracy_l=np.mean(x_train_pred==y_test)
test_accuracy_l

#checling on cross tab
pd.crosstab(x_train_pred_l,y_test)

#checking accuracy on train deta
train_predict_l=train_nb.predict(x_train)
train_accuracy_l=np.mean(y_train==train_predict)
train_accuracy_l
pd.crosstab(y_train, train_predict)
