# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 22:26:10 2019

@author: Ritzy

"""

#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix


data = pd.read_csv('Churn_Modelling.csv')

x = data.iloc[:, 3:13].values
y = data.iloc[:, 13].values

# Encoding labels 
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x_1 = LabelEncoder()
x[:, 1] = le_x_1.fit_transform(x[:, 1])
le_x_2 = LabelEncoder()
x[:, 2] = le_x_2.fit_transform(x[:, 2])

# creating dummy var
ohe= OneHotEncoder(categorical_features =[1])
x = ohe.fit_transform(x).toarray()
x = x[:, 1:]

# Splitting into training and testing data
# from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# feature scaling
# from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train= sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

#creating the NN
#from keras.models import Sequential
classifier = Sequential()

#creating layers
#from keras.layers import Dense
classifier.add(Dense(output_dim=6, init='uniform', activation ='relu', input_dim=11))
classifier.add(Dense(output_dim=6, init='uniform', activation ='relu'))
classifier.add(Dense(output_dim=1, init='uniform', activation ='sigmoid'))
 
# Compiling NN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 


# fitting our model
classifier.fit(x_train, y_train, batch_size=10, epochs=100) 

# prediction
y_pred = classifier.predict(x_test)
y_pred=(y_pred>0.5)

# creating a confusion matrix
#from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# printing confusion matrix
cm



