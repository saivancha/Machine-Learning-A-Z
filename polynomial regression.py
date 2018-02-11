# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:27:22 2018

@author: saiva
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,-1].values

"""#splitting data into training set and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)"""

"""#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)"""

#fitting linear regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

#fitting polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

#visualizing linear regression results
plt.scatter(x,y, color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Truth or False')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

#visualizing polynomial regression results
x_grid=np.arange(min(x), max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y, color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('Truth or False')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

#predicting a new result with linear regression
lin_reg.predict(6.5)

#predicting a new result with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))