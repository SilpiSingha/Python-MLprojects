# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:04:00 2019

@author: tgaga
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#reading the data from dataset
dataset = pd.read_csv('AirQuality.csv')
X = dataset.iloc[:,2:14].values
y = dataset.iloc[:,14:15].values

#removing missing values
'''from sklearn.preprocessing import Imputer
impute = Imputer(missing_values="NaN",strategy="mean",axis=0)
X = impute.fit_transform(X)
y = impute.fit_transform(y)

#splitting data into training and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = \
train_test_split(X,y,train_size = 0.85,random_state=0)


#fit the model in multiple linear regression
from sklearn.linear_model import LinearRegression
reg_lin = LinearRegression()
reg_lin.fit(X_train,y_train)

#predicting the results for linear regression
y_pred_lin = reg_lin.predict(X_test)

lin_act = sum(y_test)/len(y_test)
lin_obt = sum(y_pred_lin)/len(y_pred_lin)

perc_acc_lin = 1-((lin_act-lin_obt)/lin_act)*100

#fit the model in multiple decision tree
from sklearn.tree import DecisionTreeRegressor
reg_dt = DecisionTreeRegressor(random_state=0)
reg_dt.fit(X_train,y_train)

#predicting the results for decision tree 
y_pred_dt = reg_dt.predict(X_test)

dt_act = sum(y_test)/len(y_test)
dt_obt = sum(y_pred_dt)/len(y_pred_dt)

perc_acc_dt = 1-((dt_act-dt_obt)/dt_act)*100'''

