# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 11:52:32 2020

@author: admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv("D:\\ML\\dataset\\general_data.csv")
### Check null value in dataset
dataset_null = dataset.isnull().sum()
### replace the null values 
new_numcompaniesworked = np.where(dataset['NumCompaniesWorked'],2.69,dataset['NumCompaniesWorked'])
dataset['NumCompaniesWorked']= new_numcompaniesworked
new_totalworkingyears=np.where(dataset['TotalWorkingYears'],11.27,dataset['TotalWorkingYears'])
dataset['TotalWorkingYears']=new_totalworkingyears
## drop other features
df = dataset.drop(columns=['EmployeeCount','EmployeeID','Over18','StandardHours'])

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

##### To Model Creation 
from sklearn import tree
tree_model = tree.DecisionTreeClassifier(max_depth=38)
predictors = pd.DataFrame([df['Age'],df['BusinessTravel'],df['Department'],df['DistanceFromHome'],df['Education'],df['EducationField'],df['Gender'],df['JobLevel'],df['JobRole'],df['MaritalStatus'],df['MonthlyIncome'],df['PercentSalaryHike'],df['StockOptionLevel'],df['TrainingTimesLastYear'],df['YearsAtCompany'],df['YearsSinceLastPromotion'],df['YearsWithCurrManager'],df['NumCompaniesWorked'],df['TotalWorkingYears']]).T
tree_model.fit(X = predictors,y=df['Attrition'])
with open("Dattrition",'w') as f:
    f = tree.export_graphviz(tree_model,feature_names=['Age','BusinessTravel', 'Department', 'DistanceFromHome','Education', 'EducationField', 'Gender', 'JobLevel', 'JobRole','MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked','PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears','TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion','YearsWithCurrManager'],out_file=f)
    
###### Model Accuracy
tree_model.score(X = predictors,y = df['Attrition'])

#### Random Forest Classifer
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=1000,max_features=2,oob_score=2)
features = ['Age', 'BusinessTravel', 'Department', 'DistanceFromHome','Education', 'EducationField', 'Gender', 'JobLevel', 'JobRole','MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked','PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears','TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion','YearsWithCurrManager']
rf_model.fit(X = df[features],y=df['Attrition'])
print("OOB_Score:",rf_model.oob_score_)

#####To Find the Important variable
for features,imp in zip (features,rf_model.feature_importances_):
    print(features,imp)

##### Bulid Decision Tree Models
tree_model = tree.DecisionTreeClassifier(max_depth=10)
predictors = pd.DataFrame([df["Age"],df["MonthlyIncome"],df["YearsAtCompany"],df["DistanceFromHome"],df["PercentSalaryHike"]]).T
tree_model.fit(X = predictors,y=df['Attrition'])
with open ("Dattrition1.dot",'w') as f:
    f = tree.export_graphviz(tree_model,feature_names =['Age','MonthlyIncome','YearsAtCompany','DistanceFromHome','PercentSalaryHike'],out_file=f)

##### Model Accuracy score:
tree_model.score(X=predictors, y=df['Attrition'])

##### Bulid Decision Tree Models for Indepent variable = Age,MontlyIncome
tree_model = tree.DecisionTreeClassifier(max_depth=4)
predictors = pd.DataFrame([df["Age"],df["MonthlyIncome"]]).T
tree_model.fit(X = predictors,y=df['Attrition'])
with open ("Dattrition2.dot",'w') as f:
    f = tree.export_graphviz(tree_model,feature_names =['Age','MonthlyIncome'],out_file=f)

##### Model Accuracy score:
tree_model.score(X=predictors, y=df['Attrition'])