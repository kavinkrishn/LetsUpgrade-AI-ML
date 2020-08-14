# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 18:03:28 2020

@author: admin
"""

import pandas as pd
import numpy as sns
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("D:\\ML\\dataset\\train.csv")

le = preprocessing.LabelEncoder()
dataset['Sex'] = le.fit_transform(dataset['Sex'])
dataset['Embarked'] = le.fit_transform(dataset['Embarked'])
dataset_null = dataset.isnull().sum()
from sklearn import neighbors
y = dataset['Pclass']
X = dataset.drop(['Pclass','PassengerId','Name','Cabin','Ticket'],axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train).score(X_test,y_test)
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)

def knn(k):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    score = knn.fit(X_train,y_train).score(X_test,y_test)
    print("K = ",k,"Accuracy Score:",score)
    y_pred = knn.predict(X_test)
    confusion_matrix(y_test,y_pred)
    
for k in range(1,len(X_test)+1):
    knn(k)


