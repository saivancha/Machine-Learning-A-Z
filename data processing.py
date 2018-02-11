# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:55:53 2018

@author: saiva
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values


#splitting data into training set and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

"""#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)"""