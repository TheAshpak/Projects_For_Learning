# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 18:54:00 2021

@author: theas
"""

#importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading deta set

score = pd.read_csv("D:/DataScience/Class/assignment working/SLR/SAT_GPA.csv")

#EDA
score.describe()
#kurtoisis
score.kurt()

#skewness
score.skew()

#visualising data

#for target

#barplot

plt.bar(x= np.arange(1,201,1),height=score.GPA) 

#histigram
plt.hist(score.GPA)

#box plot
plt.boxplot(score.GPA)
plt.show() #no outliers

#for predictor

#barplot

plt.bar(x= np.arange(1,201,1),height=score.SAT_Scores) 

#histigram
plt.hist(score.SAT_Scores)

#box plot
plt.boxplot(score.SAT_Scores)
plt.show() #no outliers
#building model

#checking correlation between predictor aand target

plt.scatter(score.SAT_Scores,score.GPA)

#there is no correlation between both the variables

#finding correlation coeficient

np.corrcoef(score.SAT_Scores,score.GPA)

#finding covariance
np.cov(score.SAT_Scores,score.GPA)

#building model
import statsmodels.formula.api as smf


model_1=smf.ols('GPA ~ SAT_Scores',data=score).fit()
model_1.summary()

pred_1=model_1.predict(score.SAT_Scores)

# Regression Line
plt.scatter(score.SAT_Scores,score.GPA)
plt.plot(score.SAT_Scores, pred_1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

#calculating erroes
err_1=score.GPA - pred_1
err_sq_1 = np.square(err_1)
mean_err_1=np.mean(err_sq_1)
rmse_1= np.sqrt(mean_err_1)
rmse_1

##### Model building on Transformed Data
# Log Transformation

#scatter plot
plt.scatter(np.log(score.SAT_Scores),score.GPA)

#correlation
np.corrcoef(np.log(score.SAT_Scores),score.GPA)

#covariance
np.cov(np.log(score.SAT_Scores),score.GPA)

#building model
model_2=smf.ols('GPA ~ np.log(SAT_Scores)',data=score).fit()
model_2.summary()

#failed to reject null hypothesis

pred_2=model_2.predict(score.SAT_Scores)

# Regression Line
plt.scatter(np.log(score.SAT_Scores),score.GPA)
plt.plot(np.log(score.SAT_Scores), pred_2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

#calculating erroes
err_2=score.GPA - pred_2
err_sq_2 = np.square(err_2)
mean_err_2=np.mean(err_sq_2)
rmse_2= np.sqrt(mean_err_2)
rmse_2


### Exponential transformation

plt.scatter(score.SAT_Scores,np.log(score.GPA))

#correlation
np.corrcoef(score.SAT_Scores,np.log(score.GPA))

#covariance
np.cov(score.SAT_Scores,np.log(score.GPA))

#building model
model_3=smf.ols('np.log(score.GPA) ~ SAT_Scores',data=score).fit()
model_3.summary()

pred_3=model_3.predict(score.SAT_Scores)

# Regression Line
plt.scatter(score.SAT_Scores,np.log(score.GPA))
plt.plot(score.SAT_Scores, pred_3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

#calculating erroes
err_3=score.GPA - pred_3
err_sq_3 = np.square(err_3)
mean_err_3=np.mean(err_sq_3)
rmse_3= np.sqrt(mean_err_3)
rmse_3

# Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model_4 = smf.ols('np.log(GPA) ~ SAT_Scores + I(SAT_Scores*SAT_Scores)', data = score).fit()
model_4.summary()

pred_4 = model_4.predict(score.SAT_Scores)
pred_4_at = np.exp(pred_4)
pred_4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = score.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)



plt.scatter(score.SAT_Scores, np.log(score.GPA))
plt.plot(X, pred_4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res_4 = score.GPA - pred_4_at
res_sqr_4 = res_4 * res_4
mse_4 = np.mean(res_sqr_4)
rmse_4 = np.sqrt(mse_4)
rmse_4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse_1, rmse_2, rmse_3, rmse_4])}
table_rmse = pd.DataFrame(data)
table_rmse



