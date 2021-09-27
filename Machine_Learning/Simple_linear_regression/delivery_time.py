# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:56:45 2021

@author: theas
"""
#importing required data sets
import pandas as pd
import numpy as np

del_time=pd.read_csv("D:/DataScience/Class/assignment working/SLR/delivery_time.csv")
del_time=del_time.rename(columns=({ "Delivery Time" : "Delivery_time" , "Sorting Time" :"Sorting_time" }))
#EDA
del_time.describe()

#checking kurtoisis
del_time.kurt()

#ckecking skewness
del_time.skew()

#checking graphical representation of data
import matplotlib.pyplot as plt

plt.bar(height=del_time["Delivery_time"],x=np.arange(1,22,1))
plt.show()
plt.hist(del_time["Delivery_time"])
plt.boxplot(del_time["Delivery_time"])

#for target
plt.bar(height=del_time["Sorting_time"],x=np.arange(1,22,1))
plt.show()
plt.hist(del_time["Sorting_time"])
plt.boxplot(del_time["Sorting_time"])

#scatter plot
plt.scatter(del_time["Sorting_time"],del_time["Delivery_time"])

#correlation
np.corrcoef(del_time["Delivery_time"],del_time["Sorting_time"])
#we got 0.825 as correlation coefficient its not 0.85 but we dont have other inpute variable to do any changes so we will move ahead with 0.82

#co variance
np.cov(del_time["Delivery_time"],del_time["Sorting_time"])[0,1]
del_time.cov()

#bulding model 
import statsmodels.formula.api as smf
model= smf.ols("Delivery_time ~ Sorting_time",data=del_time).fit()

model.summary()

#predictions
pred = model.predict(del_time["Sorting_time"])

#plotting regression line
plt.scatter(del_time["Sorting_time"],del_time["Delivery_time"],color="red")
plt.plot(del_time["Sorting_time"],pred,"g")
plt.legend(["Predicted Line","Observations"])
plt.show()

#error calculations

error = del_time["Delivery_time"] - pred 
er_sq = error * error
mean_er = np.mean(er_sq)
rmse_1 =np.sqrt(mean_er)
rmse_1
#Model_building on transformed data

plt.scatter(np.log(del_time["Sorting_time"]),del_time["Delivery_time"],color="red")
np.corrcoef(np.log(del_time["Sorting_time"]), del_time["Delivery_time"]) 
#correlation
np.cov(np.log(del_time["Sorting_time"]),del_time["Delivery_time"])

model_2 = smf.ols('Delivery_time ~ np.log(Sorting_time)', data = del_time).fit()
model_2.summary()

######### p> 0.05 faild to reject null hypothesis

#model building on transformed data 
plt.scatter(np.log(del_time["Sorting_time"]),np.log(del_time["Delivery_time"]),color="red")
np.corrcoef(np.log(del_time["Sorting_time"]), np.log(del_time["Delivery_time"])) 
#correlation
np.cov(np.log(del_time["Sorting_time"]),np.log(del_time["Delivery_time"]))

model_3 = smf.ols('np.log(Delivery_time) ~ np.log(Sorting_time)', data = del_time).fit()
model_3.summary()

pred_3=model_3.predict(del_time["Sorting_time"])
#plotting regression line

plt.scatter(np.log(del_time["Sorting_time"]),np.log(del_time["Delivery_time"]),color="red")
plt.plot(np.log(del_time["Sorting_time"]),pred_3,"g")
plt.legend(["Predicted Line","Observations"])
plt.show()

#error calculations

error_3 = del_time["Delivery_time"] - pred_3
er_sq_3 = error_3 * error_3
mean_er_3 = np.mean(er_sq_3)
rmse_3 =np.sqrt(mean_er_3)
rmse_3

#### Polynomial transformation
# x = Sorting time; x^2 = Sorting_time*Sorting_time; y = log(Delivery time)

model_4 = smf.ols('np.log(Delivery_time) ~ Sorting_time + I(Sorting_time * Sorting_time)', data = del_time).fit()

model_4.summary()

####polynominal failed to reject null hypothesis
