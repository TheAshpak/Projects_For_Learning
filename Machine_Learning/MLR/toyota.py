# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:35:12 2021

@author: theas
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

#importing data set

toyota=pd.read_csv("D:/DataScience/Class/Assignments/MLR/ToyotaCorolla.csv")

#EDA
toyota.shape

toyota.info()
x=["Price", "Age_08_04", "KM", "HP", "cc", "Doors", "Gears", "Quarterly_Tax", "Weight"]

car=toyota.loc[:,x]
car.describe()

#bar plots
plt.figure()
car.plot(figsize=(30,30),kind="bar",subplots=True,layout=(6,4))
plt.show()

# histogram
plt.figure()
car.plot(figsize=(30,30),kind="hist",subplots=True,layout=(6,6))
plt.show()

#box plot

car.plot(kind="box",figsize=(15,15),subplots=True,layout=(4,3))
plt.show()

import seaborn as sns

#checking null values
car.isna().sum()

#winsorization

from feature_engine.outliers import Winsorizer

win=Winsorizer(capping_method ="iqr",fold=1.5,tail='both')
car=win.fit_transform(car)

#box plot

car.plot(kind="box",figsize=(15,15),subplots=True,layout=(4,3))
plt.show()

#checking duplicates
car.duplicated().sum()
car.drop_duplicates(keep="first",inplace=True)

# Jointplot
import seaborn as sns
sns.jointplot(x=car.Price, y=car.HP)

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(car['HP'])

# Q-Q Plot
from scipy import stats
import pylab
for i in car.columns:
    stats.probplot(car[i], dist = "norm", plot = pylab)
    plt.show()


# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(car)

sns.heatmap(car.corr())


car.plot(kind="hist",subplots=True,layout=(4,4),figsize=(15,15))

car.describe()

car.info()
car.corr()
x=car.drop("Price",axis=1)
y=car.Price
from sklearn.feature_selection import f_regression
from sklearn.model_selection import  RepeatedStratifiedKFold ,cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

scale= StandardScaler()
x_scale=scale.fit_transform(x)
fs= SelectKBest(score_func=f_regression,k="all")
fs.fit(x_scale,y)
fs.scores_

plt.bar([i for i in range(len(fs.scores_))],fs.scores_)
plt.show()

fs_1= SelectKBest(score_func=f_regression,k=5)
#fs_1.fit(x_scale,y)
#fs_1.n_features_in_
#print([i for i in fs_1.scores_])

pipeline=Pipeline(steps=[("fs",fs_1),("Lr",LinearRegression())])

cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
scores = cross_val_score(pipeline,x,y , cv=cv ,n_jobs=-1,scoring='r2')

scores.mean()
scores.std()


