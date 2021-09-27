# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:05:27 2021

@author: theas
"""

#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading data set
cal= pd.read_csv("D:/DataScience/Class/assignment working/SLR/calories_consumed.csv")
cal=cal.rename(columns={"Weight gained (grams)" : "weight_gain" , "Calories Consumed" : "cal_con"})
#EDA

cal.describe()

#visualizing and checking data

plt.bar(height=cal.cal_con, x=np.arange(1,15,1))
plt.boxplot(cal.cal_con) #no outliers
plt.hist(cal.cal_con)

plt.bar(height=cal.weight_gain, x=np.arange(1,15,1))
plt.boxplot(cal.weight_gain) #no outliers
plt.hist(cal.weight_gain)

#building model
plt.scatter(cal.weight_gain,cal.cal_con,color="red")

#correlation
np.corrcoef(cal.cal_con,cal.weight_gain)

#covarianc
np.cov(cal.cal_con,cal.weight_gain)

import statsmodels.formula.api as smf
model_1=smf.ols('cal_con ~ weight_gain',data=cal).fit()
model_1.summary()

pred_1=model_1.predict(cal.weight_gain)

#plotting regression line
plt.scatter(cal.weight_gain,cal.cal_con,color="red")
plt.plot(cal.weight_gain,pred_1)
plt.legend(["Predicted_line","Observed data"])
plt.show()

#calculating error
err_1= cal.cal_con - pred_1
err_sq_1 = err_1 * err_1 
err_mean_1 = np.mean(err_sq_1)
rmse_1 = np.sqrt(err_mean_1) 
rmse_1

#logarithmic model
#building model
#log of predictor
plt.scatter(np.log(cal.weight_gain),cal.cal_con,color="red")

#correlation
np.corrcoef(cal.cal_con,np.log(cal.weight_gain))

#covarianc
np.cov(cal.cal_con,np.log(cal.weight_gain))
    
import statsmodels.formula.api as smf
model_2=smf.ols('cal_con ~ np.log(weight_gain)',data=cal).fit()
model_2.summary()

pred_2=model_2.predict(cal.weight_gain)

#plotting regression line
plt.scatter(np.log(cal.weight_gain),cal.cal_con,color="red")
plt.plot(np.log(cal.weight_gain),pred_2)
plt.legend(["Predicted_line","Observed data"])
plt.show()

#calculating error
err_2= cal.cal_con - pred_2
err_sq_2 = err_2 * err_2 
err_mean_2 = np.mean(err_sq_2)
rmse_2 = np.sqrt(err_mean_2) 
rmse_2

#exponential transformation
#building model
plt.scatter(cal.weight_gain,np.log(cal.cal_con),color="red")

#correlation
np.corrcoef(np.log(cal.cal_con),cal.weight_gain)

#covarianc
np.cov(np.log(cal.cal_con),cal.weight_gain)


model_3=smf.ols('np.log(cal_con) ~ weight_gain',data=cal).fit()
model_3.summary()

pred_3=model_3.predict(cal.weight_gain)

#plotting regression line
plt.scatter(cal.weight_gain,np.log(cal.cal_con),color="red")
plt.plot(cal.weight_gain,pred_3)
plt.legend(["Predicted_line","Observed data"])
plt.show()

#calculating error
err_3= cal.cal_con - pred_3
err_sq_3 = err_3 * err_3 
err_mean_3 = np.mean(err_sq_3)
rmse_3 = np.sqrt(err_mean_3) 
rmse_3

#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model_4 = smf.ols('np.log(cal.cal_con) ~ weight_gain + I(weight_gain*weight_gain)', data = cal).fit()
model_4.summary()

#we failed to reject null hypothesis

pred_4 = model_4.predict(pd.DataFrame(cal))
pred_4_at = np.exp(pred_4)
pred_4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = cal.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(cal.weight_gain, np.log(cal.cal_con))
plt.plot(X, pred_4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res_4 = cal.cal_con - pred_4_at
res_sqr_4 = res_4 * res_4
mse_4 = np.mean(res_sqr_4)
rmse_4 = np.sqrt(mse_4)
rmse_4

data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse_1, rmse_2, rmse_3, rmse_4])}
table_rmse = pd.DataFrame(data)
table_rmse
