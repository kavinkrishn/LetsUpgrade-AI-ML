# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 20:03:54 2020

@author: admin
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
dataset = pd.read_csv("D:\\ML\\dataset\\train.csv")

le = preprocessing.LabelEncoder()
dataset['Sex'] = le.fit_transform(dataset['Sex'])
dataset['Embarked'] = le.fit_transform(dataset['Embarked'])
dataset_null = dataset.isnull().sum()

from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import svm
df = dataset.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
X = df['Pclass']
y = df.drop(['Pclass'],axis=1)
df['Age'] = df['Age'].astype(int)
df['Fare'] = df['Fare'].astype(int)

from sklearn.model_selection import train_test_split

def svmmodel(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
    clf = svm.SVC(gamma=0.01,C=100)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    a = accuracy_score(y_test,y_pred,normalize=True)
    return a
y1 = df["Pclass"]
X1 = df[['Survived','Sex', 'Age', 'SibSp','Parch','Fare','Embarked']]
svmmodel(X1,y1)

y1 = df['Survived']
X1 = df[['Pclass','Sex', 'Age', 'SibSp','Parch','Fare','Embarked']]
svmmodel(X1,y1)

y1 = df['Sex']
X1 = df[['Pclass','Survived', 'Age', 'SibSp','Parch','Fare','Embarked']]
svmmodel(X1,y1)

y1 = df['Age']
X1 = df[['Pclass','Survived', 'Sex', 'SibSp','Parch','Fare','Embarked']]
svmmodel(X1,y1)

y1 = df['SibSp']
X1 = df[['Pclass','Survived', 'Sex','Age' ,'Parch','Fare','Embarked']]
svmmodel(X1,y1)

y1 = df['Parch']
X1 = df[['Pclass','Survived', 'Sex','Age' ,'SibSp','Fare','Embarked']]
svmmodel(X1,y1)

y1 = df['Fare']
X1 = df[['Pclass','Survived', 'Sex','Age' ,'SibSp','Parch','Embarked']]
svmmodel(X1,y1)

y1 = df['Embarked']
X1 = df[['Pclass','Survived', 'Sex','Age' ,'SibSp','Fare','Parch']]
svmmodel(X1,y1)