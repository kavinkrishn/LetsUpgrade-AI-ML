# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import preprocessing 
titanic_train = pd.read_csv("D:\\ML\\dataset\\train.csv")
## to chack the null values
titanic_train_null = titanic_train.isnull().sum()
titanic_train['Cabin']=titanic_train['Cabin'].mode()
lable_encoder = preprocessing.LabelEncoder()
encoded_sex = lable_encoder.fit_transform(titanic_train['Sex'])

###### To Model Creation

tree_model = tree.DecisionTreeClassifier()
tree_model.fit(X=pd.DataFrame(encoded_sex),y=pd.DataFrame(titanic_train['Survived']))
with open("dtree.dot",'w') as f:
    f=tree.export_graphviz(tree_model,feature_names=['Sex'],out_file=f)
    
###### add more independent variable 
predictors = pd.DataFrame([encoded_sex,titanic_train['Age'],titanic_train['Pclass'],titanic_train['Fare']]).T
tree_model = tree.DecisionTreeClassifier(max_depth=8)
tree_model.fit(X=predictors,y=titanic_train['Survived'])

with open("Dtree1.dot",'w') as f:
    f = tree.export_graphviz(tree_model,feature_names=["Sex","Age","Pclass","Fare"],out_file=f)

###### Model Accuracy
tree_model.score(X=predictors,y=titanic_train['Survived'])

###### Decision Tree helpful for predicition
titanic_test = pd.read_csv("D:\\ML\\dataset\\test.csv")
titanic_test_null = titanic_test.isnull().sum()
encode_sex_test = lable_encoder.fit_transform(titanic_test['Sex'])
test_features = pd.DataFrame([encode_sex_test,titanic_test['Pclass'],titanic_test['Age'],titanic_test['Fare']]).T
#test_preds = tree_model.predict(X=pd.DataFrame([encode_sex_test,titanic_test['Pclass'],titanic_test['Age'],titanic_test['Fare']]))
#predicted_output = pd.DataFrame({"Passenger_ID":titanic_test["PassengerId"],"Survive":test_preds})
# predicted_output.to_csv("output.csv",index=False)


#### RandomForestClassifer

from sklearn.ensemble import RandomForestClassifier
lable_encoder = preprocessing.LabelEncoder()
titanic_train['Sex'] = lable_encoder.fit_transform(titanic_train["Sex"])
titanic_train['Embarked'] = lable_encoder.fit_transform(titanic_train["Embarked"])
rf_model = RandomForestClassifier(n_estimators=1000,max_features=2,oob_score=True)
feature = ["Sex","Pclass","SibSp","Age","Fare","Embarked"]
rf_model.fit(X = titanic_train[feature],y=titanic_train["Survived"])
print("OOB Accuracy:",rf_model.oob_score_)

## To Find important Variable
for feature,imp in zip(feature,rf_model.feature_importances_):
    print(feature,imp)
    
####    Build the Decesion Tree
tree_model = tree.DecisionTreeClassifier()
predictors = pd.DataFrame([encoded_sex,titanic_train['Age'],titanic_train['Fare']]).T
tree_model = tree.DecisionTreeClassifier(max_depth=8)
tree_model.fit(X=predictors,y=titanic_train['Survived'])
with open("Dtree3.dot",'w') as f:
    f = tree.export_graphviz(tree_model,feature_names=["Sex","Age","Fare"],out_file=f)

###### Model Accuracy
tree_model.score(X=predictors,y=titanic_train['Survived'])