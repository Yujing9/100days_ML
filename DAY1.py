import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
#STEP1
data = pd.read_csv("./datasets/Data.csv")
# With slice objects,left is row,right is colomn
#STEP2
X = data.iloc[ : , :-1].values
Y = data.iloc[ : , 3].values
#STEP3
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.fit_transform(X[:,1:3])
#STEP4
le = preprocessing.LabelEncoder()
le.fit(Y[:])
Y[:] = le.fit_transform(Y[:])
#STEP5
X_train,X_test,Y_train,Y_test= train_test_split(X, Y, test_size=0.2, random_state=0)
#STEP6
scaler = StandardScaler()
X_train[:,1:] = scaler.fit_transform(X_train[:,1:]) 
X_test[:,1:] = scaler.fit_transform(X_test[:,1:]) 