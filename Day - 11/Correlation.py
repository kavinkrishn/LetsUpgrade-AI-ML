# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:00:03 2020

@author: admin
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read dataset and display
dataset = pd.read_csv("general_data.csv")
dataset.head()

# check null dataset
dataset.isnull().sum()

# drop null values in dataset
df = dataset.dropna()
data1 = df.isnull().sum()

# Replace Attrition values yes = 0 and no = 1
df['Attrition'].replace({"Yes" : 1,"No" : 0},inplace=True)

# Replace Gender values male = 0 and female = 1
df['Gender'].replace({"Male":0,"Female":1},inplace=True)

# Replace MaritalStatus values us Married = 0, single= 1, divorced= 2
df['MaritalStatus'].replace({"Married": 0, "Single" : 1,"Divorced": 2}, inplace= True)

# Correlation
from scipy.stats import pearsonr
# Correlation of Attrition and Age 
stats,p = pearsonr(df.Attrition,df.Age)
print(stats,p)
plt.scatter(df.Attrition,df.Age)
plt.show()

# Correlation of Attrition and Distancefromhome 
stats,p = pearsonr(df.Attrition,df.DistanceFromHome)
print(stats,p)
plt.scatter(df.Attrition,df.DistanceFromHome)
plt.show()

# Correlation of Attrition and Education 
stats,p = pearsonr(df.Attrition,df.Education)
print(stats,p)
plt.scatter(df.Attrition,df.Education)
plt.show()

# Correlation of Attrition and job level 
stats,p = pearsonr(df.Attrition,df.JobLevel)
print(stats,p)
plt.scatter(df.Attrition,df.JobLevel)
plt.show()

# Correlation of Attrition and Gender
stats,p = pearsonr(df.Attrition,df.Gender)
print(stats,p)
plt.scatter(df.Attrition,df.Gender)
plt.show()

# Correlation of Attrition and MonthlyIncome
stats,p = pearsonr(df.Attrition,df.MonthlyIncome)
print(stats,p)
plt.scatter(df.Attrition,df.MonthlyIncome)
plt.show()


# Correlation of Attrition and NumCompaniesWorked
stats,p = pearsonr(df.Attrition,df.NumCompaniesWorked)
print(stats,p)
plt.scatter(df.Attrition,df.NumCompaniesWorked)
plt.show()

# Correlation of Attrition and PercentSalaryHike
stats,p = pearsonr(df.Attrition,df.PercentSalaryHike)
print(stats,p)
plt.scatter(df.Attrition,df.PercentSalaryHike)
plt.show()

# Correlation of Attrition and TotalWorkingYears
stats,p = pearsonr(df.Attrition,df.TotalWorkingYears)
print(stats,p)
plt.scatter(df.Attrition,df.TotalWorkingYears)
plt.show()

# Correlation of Attrition and YearsAtCompany
stats,p = pearsonr(df.Attrition,df.YearsAtCompany)
print(stats,p)
plt.scatter(df.Attrition,df.YearsAtCompany)
plt.show()

# Correlation Matrix and Visualization
corr_mat = df[['Age','DistanceFromHome','Education','Gender',
       'JobLevel','MaritalStatus', 'MonthlyIncome','NumCompaniesWorked','PercentSalaryHike', 
       'TotalWorkingYears', 'YearsAtCompany','Attrition']].copy()
corr = corr_mat.corr()

import seaborn as sns
plt.figure(figsize=(12,10),dpi=100)
sns.heatmap(corr_mat.corr(),
            cmap='coolwarm',
            annot=True,
            square=True,
            robust=True);
top_corr = corr_mat.nlargest(5,"Attrition")