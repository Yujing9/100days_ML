# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 11:51:34 2022

@author: 啊元
"""
import numpy as np
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
dataset = pd.read_csv("./datasets/Social_Network_Ads.csv")
X = dataset.iloc[:,2:4].values
Y = dataset.iloc[:,4].values
X_train,X_test,Y_train,Y_test= train_test_split(X, Y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
cf = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
cf.fit(X_train, Y_train)
Y_pred = cf.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
