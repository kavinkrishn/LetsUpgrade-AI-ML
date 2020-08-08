# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:33:02 2020

@author: admin
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv("D:\ML\dataset\general_data.csv")

# check null values
dataset_null=dataset.isnull().sum()
dataset = dataset.fillna(dataset[['NumCompaniesWorked','TotalWorkingYears']].mean())

## drop other features
df = dataset.drop(columns=['EmployeeCount', 'EmployeeID','Over18','StandardHours'])

## Label Encoding technique used to change values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])
df['BusinessTravel'] = le.fit_transform(df['BusinessTravel'])
df['Department'] = le.fit_transform(df['Department'])
df['EducationField'] = le.fit_transform(df['EducationField'])
df['Gender'] = le.fit_transform(df['Gender'])
df['JobRole'] = le.fit_transform(df['JobRole'])
df['MaritalStatus'] = le.fit_transform(df['MaritalStatus'])

## Variable Selection
y = df['Attrition']
X = df.drop(columns=['Attrition'])

## Logistic Regression
import statsmodels.api as sm

X1 = sm.add_constant(X)
Logistic = sm.Logit(y,X1)
result = Logistic.fit()
result.summary()