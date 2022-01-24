# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 09:40:09 2022

@author: 啊元
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
dataset = pd.read_csv("./datasets/50_Startups.csv")
X = dataset.iloc[:,:4].values
Y = dataset.iloc[:,4:].values
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer.fit(X[:,0:3])
X[:,0:3] = imputer.transform(X[:,0:3])
le = LabelEncoder()
X[:,3] = le.fit_transform(X[:,3])   
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
X = X[: , 1:]
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
X_train,X_test,Y_train,Y_test= train_test_split(X, Y, test_size=0.2, random_state=0)
reg = LinearRegression().fit(X_train, Y_train)
Y_predi = reg.predict(X_test)
plt.plot((Y_test.min(),Y_test.max()),(Y_test.min(),Y_test.max()),color="blue")
plt.scatter(Y_test,Y_predi,color='red')
plt.xlabel("Y_test");plt.ylabel("Y_pred")
plt.style.use('ggplot')
plt.show()