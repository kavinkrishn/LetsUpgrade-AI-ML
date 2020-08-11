# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:12:20 2020

@author: admin
"""

import pandas as pd
dataset = pd.read_csv("D:\\ML\\dataset\\train.csv")

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix

le = preprocessing.LabelEncoder()
dataset['Sex'] = le.fit_transform(dataset['Sex'])
dataset['Embarked'] = le.fit_transform(dataset['Embarked'])

y = dataset['Survived'] 
X = dataset.drop(['Survived','Name','PassengerId','Cabin','Ticket'],axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn.naive_bayes import *
clf = BernoulliNB()
y_pred = clf.fit(X_train,y_train).predict(X_test)
accuracy_score(y_test,y_pred,normalize=True)
confusion_matrix(y_test,y_pred)

def titanic_model (X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    clf = BernoulliNB()
    y_pred = clf.fit(X_train,y_train).predict(X_test)
    print(accuracy_score(y_test,y_pred,normalize=True))
    print(confusion_matrix(y_test,y_pred))
    
dataset['Age'] = dataset['Age'].astype(int)
dataset.loc[dataset['Age'] <= 22, 'Age'] = 0
dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 32), 'Age'] = 1
dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 45), 'Age'] = 2
dataset.loc[(dataset['Age'] > 45) & (dataset['Age'] <= 80), 'Age'] = 3
    
X = dataset.drop(['Age','Name','PassengerId','Cabin','Ticket'],axis=1)
y = dataset['Age']
titanic_model(X,y)

dataset['Fare'] = dataset['Fare'].astype(int)
dataset.loc[dataset['Fare'] <=7,'Fare'] = 0
dataset.loc[(dataset['Fare'] > 7) & (dataset['Fare'] <=14)] = 1
dataset.loc[(dataset['Fare'] > 14) & (dataset['Fare'] <=31)] = 2
dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <=512)] = 3
X = dataset.drop(['Fare','Name','PassengerId','Cabin','Ticket'],axis=1)
y = dataset['Fare']
titanic_model(X,y)

