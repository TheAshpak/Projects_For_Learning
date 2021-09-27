# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:49:51 2021

@author: theas
"""
#importing required libraries
import pandas as pd
import numpy as np

salary=pd.read_csv("D:/DataScience/Class/assignment working/SLR/Salary_Data.csv")

#EDA

salary.describe()

#Visualisation

import matplotlib.pyplot as plt 

plt.bar(height = salary.YearsExperience, x = np.arange(1, 31, 1))
plt.hist(salary.YearsExperience) 
plt.boxplot(salary.YearsExperience) 

plt.bar(height = salary.Salary, x = np.arange(1, 31, 1))
plt.hist(salary.Salary) 
plt.boxplot(salary.Salary)

# Scatter plot
plt.scatter(x = salary["Salary"], y = salary['YearsExperience'], color = 'green') 

# correlation
np.corrcoef(salary["Salary"], salary['YearsExperience']) 

# Covariance

cov_output = np.cov(salary.Salary, salary.YearsExperience)[0, 1]
cov_output


# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('Salary ~ YearsExperience', data = salary).fit()
model.summary()

pred1 = model.predict(salary.YearsExperience)

# Regression Line
plt.scatter(salary.YearsExperience, salary.Salary)
plt.plot(salary.YearsExperience, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = salary.Salary - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

# Model building on Transformed Data
# Log Transformation

plt.scatter(np.log(salary.YearsExperience), salary.Salary)

np.corrcoef(salary.Salary, np.log(salary.YearsExperience)) #correlation

model2 = smf.ols('Salary ~ np.log(YearsExperience)', data = salary).fit()
model2.summary()

pred2 = model2.predict(salary.YearsExperience)

# Regression Line
plt.scatter(np.log(salary.YearsExperience), salary.Salary)
plt.plot(np.log(salary.YearsExperience), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = salary.Salary - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


# Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = salary["YearsExperience"], y = np.log(salary['Salary']), color = 'orange')
np.corrcoef(salary["YearsExperience"], np.log(salary.Salary)) #correlation

model3 = smf.ols('np.log(Salary) ~ YearsExperience', data = salary).fit()
model3.summary()

pred3 = model3.predict(salary.YearsExperience)
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(salary.YearsExperience, np.log(salary.Salary))
plt.plot(salary.YearsExperience, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = salary.Salary - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

# Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(Salary) ~ YearsExperience + I(YearsExperience*YearsExperience)', data = salary).fit()
model4.summary()

pred4 = model4.predict(salary.YearsExperience)
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = salary.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)



plt.scatter(salary.YearsExperience, np.log(salary.Salary))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = salary.Salary - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

