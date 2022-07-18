# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 14:41:42 2022

@author: sayan
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Implementing the Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


#Implementing the Polynomial Linear Regression Feature
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


#visualising the linear regression
plt.scatter( X, y, color= 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')

#visualising the polynomial regression
plt.scatter(X, y , color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = "blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')


#predicting new result using the linear regression
lin_reg.predict([[6.5]])

#predicting new result using the polynomial regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))