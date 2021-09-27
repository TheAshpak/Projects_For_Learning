# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 18:18:06 2021

@author: theas
"""

#importing required pacckages

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#loaading data

s_u=pd.read_csv("D:/DataScience/Class/assignment working/MLR/50_Startups.csv")

#renaming columns
s_u.columns=s_u.columns.str.replace(" ","_")
s_u.columns=s_u.columns.str.replace("&","_")
#EDA

s_u.describe()

#checking head
s_u.head(10)

#type casting

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()

s_u.State=lb.fit_transform(s_u["State"])

#handling missing values 

s_u["R_D_Spend"]=s_u["R_D_Spend"].replace(0 ,np.NAN).fillna(method=("ffill"))

s_u["Marketing_Spend"]=s_u["Marketing_Spend"].replace(0 ,np.NaN).fillna(method=("ffill"))

#checking duplicates
s_u.duplicated().sum()

#no duplicates 

#checking outliers

s_u.plot(kind="box",figsize=(15,15),subplots=True,layout=(4,3))
plt.show()

#no outliers

#normalizing data
'''
def norm(x):
    z=(x-x.min())/(x.max()-x.min())
    return z
norm_su=norm(s_u)               
'''
#Graphical representation

# Jointplot
import seaborn as sns
sns.jointplot(x=s_u["Marketing_Spend"], y=s_u["Profit"])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(s_u["Profit"])

plt.figure(1, figsize=(16, 10))
sns.countplot(s_u["Marketing_Spend"])


# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(s_u["Marketing_Spend"],dist="norm",plot=pylab)
plt.show()

stats.probplot(s_u["R_D_Spend"],dist="norm",plot=pylab)
plt.show()

stats.probplot(s_u["Administration"],dist="norm",plot=pylab)
plt.show()

stats.probplot(s_u["State"],dist="norm",plot=pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(s_u)
plt.show()

#checking correlation
s_u.corr()

#building model

import statsmodels.formula.api as smf

ml1 = smf.ols('Profit ~ R_D_Spend + Administration + Marketing_Spend + State',data=s_u).fit()

ml1.summary()

# Influence Index Plots
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)

#checking if those feature which have p value more than 0.05 have any influencial values

import statsmodels.api as sm
plt.figure(figsize=(15,15))
sm.graphics.plot_partregress_grid(ml1)
plt.show()

#droppong influencial values
new_start=s_u.drop([49,47],axis=0 )

ml_new=smf.ols('Profit ~ R_D_Spend + Administration + Marketing_Spend + State',data=new_start).fit()
ml_new.summary()

#still p value is hight
#now we have to take a call on droping feature

#calculating VIF

rqur_r_d=smf.ols('R_D_Spend ~ Administration + Marketing_Spend + State',data=s_u).fit().rsquared
vif_rd=1/(1-rqur_r_d)
vif_rd

rqur_adm=smf.ols('Administration  ~ R_D_Spend + Marketing_Spend + State',data=s_u).fit().rsquared
vif_adm=1/(1-rqur_adm)
vif_adm

rqur_mar=smf.ols(' Marketing_Spend ~ Administration + R_D_Spend + State',data=s_u).fit().rsquared
vif_mar=1/(1-rqur_mar)
vif_mar

rqur_st=smf.ols(' State ~ Marketing_Spend + Administration + R_D_Spend ',data=s_u).fit().rsquared
vif_st=1/(1-rqur_st)
vif_st

dic={"variable":["State", "Marketing_Spend" , "Administration" ,"R_D_Spend"],"VIF":[vif_st,vif_mar,vif_adm,vif_rd]}
vif=pd.DataFrame(dic)
vif
#trying with droping R&d
f_model=smf.ols('Profit ~ Administration + Marketing_Spend + State',data=new_start).fit()
f_model.summary()

sm.graphics.plot_partregress_grid(f_model)

f2_model=smf.ols('Profit ~ R_D_Spend + Administration + State',data=new_start).fit()
f2_model.summary()
sm.graphics.plot_partregress_grid(f2_model)

f3_model=smf.ols('Profit ~ R_D_Spend + Administration + Marketing_Spend',data=new_start).fit()
f3_model.summary()
sm.graphics.plot_partregress_grid(f3_model)

f4_model=smf.ols('Profit ~ R_D_Spend + State+ Marketing_Spend',data=new_start).fit()
f4_model.summary()
sm.graphics.plot_partregress_grid(f4_model)

