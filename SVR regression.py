# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:34:58 2018

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

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)


#fitting regression model to the dataset
#create your regressor here
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)

#predicting a new result
y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

#visualizing SVR results
plt.scatter(x,y, color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Truth or False')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

