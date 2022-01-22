# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 14:47:35 2022

@author: 啊元
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#data processing
data = pd.read_csv(".\datasets\studentscores.csv")
X = data.iloc[:,:1].values
Y = data.iloc[:,1].values
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#step2
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
#STEP3
Y_pred = regressor.predict(X_train)
#STEP4
plt.scatter(X_train,Y_train, color = 'red')#绘制散点图
plt.plot(X_train,Y_pred, color ='blue')#绘制线条
plt.scatter(X_test,Y_test, color = 'red')#绘制散点图
plt.plot(X_test,regressor.predict(X_test), color ='blue')#绘制线条