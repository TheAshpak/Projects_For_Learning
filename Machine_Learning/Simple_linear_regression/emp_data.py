# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 17:18:50 2021

@author: theas
"""

# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading data set
churn=pd.read_csv("D:/DataScience/Class/assignment working/SLR/emp_data.csv")

#EDA

churn.describe()

#visualizing data 
#for target
 
#bar plot
plt.bar(height=churn.Churn_out_rate ,x= np.arange(1,11,1))
#histogram
plt.hist(churn.Churn_out_rate)
#boxplot
plt.boxplot(churn.Churn_out_rate)

#for predictor 
#bar plot
plt.bar(height=churn.Salary_hike ,x= np.arange(1,11,1))
#histogram
plt.hist(churn.Salary_hike)
#boxplot
plt.boxplot(churn.Salary_hike)

#scatter plot
plt.scatter(churn.Salary_hike,churn.Churn_out_rate)

#Simple linear model
import statsmodels.formula.api as smf

#correlation
np.corrcoef(churn.Salary_hike,churn.Churn_out_rate)

#covariance
np.cov(churn.Salary_hike,churn.Churn_out_rate)

#building model
model_1=smf.ols('Churn_out_rate ~ Salary_hike',data=churn).fit()
model_1.summary()

pred_1=model_1.predict(churn.Salary_hike)

# Regression Line
plt.scatter(churn.Salary_hike,churn.Churn_out_rate)
plt.plot(churn.Salary_hike, pred_1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

#calculating erroes
err_1=churn.Churn_out_rate - pred_1
err_sq_1 = np.square(err_1)
mean_err_1=np.mean(err_sq_1)
rmse_1= np.sqrt(mean_err_1)
rmse_1

##### Model building on Transformed Data
# Log Transformation

#scatter plot
plt.scatter(np.log(churn.Salary_hike),churn.Churn_out_rate)

#correlation
np.corrcoef(np.log(churn.Salary_hike),churn.Churn_out_rate)

#covariance
np.cov(np.log(churn.Salary_hike),churn.Churn_out_rate)

#building model
model_2=smf.ols('Churn_out_rate ~ np.log(Salary_hike)',data=churn).fit()
model_2.summary()

pred_2=model_2.predict(churn.Salary_hike)

# Regression Line
plt.scatter(np.log(churn.Salary_hike),churn.Churn_out_rate)
plt.plot(np.log(churn.Salary_hike), pred_2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

#calculating erroes
err_2=churn.Churn_out_rate - pred_2
err_sq_2 = np.square(err_2)
mean_err_2=np.mean(err_sq_2)
rmse_2= np.sqrt(mean_err_2)
rmse_2


### Exponential transformation

plt.scatter(churn.Salary_hike,np.log(churn.Churn_out_rate))

#correlation
np.corrcoef(churn.Salary_hike,np.log(churn.Churn_out_rate))

#covariance
np.cov(churn.Salary_hike,np.log(churn.Churn_out_rate))

#building model
model_3=smf.ols('np.log(Churn_out_rate) ~ Salary_hike',data=churn).fit()
model_3.summary()

pred_3=model_3.predict(churn.Salary_hike)

# Regression Line
plt.scatter(churn.Salary_hike,np.log(churn.Churn_out_rate))
plt.plot(churn.Salary_hike, pred_3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

#calculating erroes
err_3=churn.Churn_out_rate - pred_3
err_sq_3 = np.square(err_3)
mean_err_3=np.mean(err_sq_3)
rmse_3= np.sqrt(mean_err_3)
rmse_3

# Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model_4 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data = churn).fit()
model_4.summary()

pred_4 = model_4.predict(churn.Salary_hike)
pred_4_at = np.exp(pred_4)
pred_4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = churn.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)



plt.scatter(churn.Salary_hike, np.log(churn.Churn_out_rate))
plt.plot(X, pred_4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res_4 = churn.Churn_out_rate - pred_4_at
res_sqr_4 = res_4 * res_4
mse_4 = np.mean(res_sqr_4)
rmse_4 = np.sqrt(mse_4)
rmse_4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse_1, rmse_2, rmse_3, rmse_4])}
table_rmse = pd.DataFrame(data)
table_rmse

