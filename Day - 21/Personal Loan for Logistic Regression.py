# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:34:30 2020

@author: admin
"""

import pandas as pd
import numpy as np

dataset = pd.read_excel('D:\\ML\\dataset\\Bank_Personal_Loan_Modelling.xlsx',sheet_name=1)

# check the null values
dataset_null= dataset.isnull().sum()

dataset = dataset.drop(["ID","ZIP Code"],axis=1)

dataset1 = dataset[['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education',
       'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard','Personal Loan']]

#independent and dependent Features
X = dataset1.iloc[:,:-1]  # independent Features
y = dataset1.iloc[:,-1]  # dependet Feature

# multicollinearity
X.iloc[:,:-1].corr()

import statsmodels.api as sm
X1 = sm.add_constant(X)
Logistic = sm.Logit(y,X1)
result = Logistic.fit()

result.summary()