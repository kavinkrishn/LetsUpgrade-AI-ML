# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 10:16:49 2020

@author: admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
###### Load data set
dataset = pd.read_excel("D:\\ML\\dataset\\Bank_Personal_Loan_Modelling.xlsx",sheet_name=1)

#### Check null value in dataset
dataset_null = dataset.isnull().sum()
#### Drop Unwanted Features
dataset = dataset.drop(["ID","ZIP Code"],axis=1)

##### To Model Creation
from sklearn import tree
tree_model = tree.DecisionTreeClassifier(max_depth=22)
predictors = pd.DataFrame([dataset['Age'],dataset['Experience'],dataset['Income'],dataset['Family'],dataset['CCAvg'],dataset['Education'],dataset['Mortgage'],dataset['Securities Account'],dataset['CD Account'],dataset['Online'],dataset['CreditCard']]).T
tree_model.fit(X = predictors,y=dataset['Personal Loan'])
with open ("Dbank.dot",'w') as f:
    f = tree.export_graphviz(tree_model,feature_names=['Age','Experience','Income','Family','CCAvg','Education','Mortgage','Securities Account','CD Account','Online','CreditCard'],out_file=f)

###### Model Accuracy
tree_model.score(X = predictors,y = dataset['Personal Loan'])

###### Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=1000,max_features=2,oob_score=2)
features = ['Age','Experience','Income','Family','CCAvg','Education','Mortgage','Securities Account','CD Account','Online','CreditCard']
rf_model.fit(X = dataset[features],y=dataset['Personal Loan'])
print("OOB_Score:",rf_model.oob_score_)

#####To Find the Important variable
for features,imp in zip (features,rf_model.feature_importances_):
    print(features,imp)

###### To make Decision tree for independent variable = ”Income”,”CCAvg”,”Education”, Dependent Variable = “Personal Loan”.
### Bulid the Tree Model
tree_model = tree.DecisionTreeClassifier()
predictors = pd.DataFrame([dataset['Income'],dataset['CCAvg'],dataset['Education']]).T
tree_model = tree.DecisionTreeClassifier(max_depth=6)
tree_model.fit(X = predictors, y = dataset['Personal Loan'])
with open ("Dbank1.dot",'w') as f:
    f = tree.export_graphviz(tree_model,feature_names=['Income','CCAvg','Education'],out_file=f)

##### Model Accuracy score:
tree_model.score(X=predictors, y=dataset['Personal Loan'])