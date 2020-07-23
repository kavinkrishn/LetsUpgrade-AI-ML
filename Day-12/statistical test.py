# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:51:57 2020

@author: admin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read the data set
dataset = pd.read_csv("general_data.csv")
dataset.head()

dataset['Attrition'].replace({'Yes':0,'No':1},inplace=True)

# Wilcoxon test
# Ho = There is no significant difference NumCompaniesWorked and Attrition
# Ha = There is  significant difference NumCompaniesWorked and Attrition
from scipy.stats import wilcoxon
stats,p = wilcoxon(dataset.Attrition,dataset.NumCompaniesWorked)
print(stats,p)

# Friedman  chi square Test
# Ho = There is no significant difference NumCompaniesWorked and Attrition and Years Since Last Promotion
# Ha = There is  significant difference NumCompaniesWorked and Attrition and Years Since Last Promotion
from scipy.stats import friedmanchisquare
p,stats = friedmanchisquare(dataset.YearsSinceLastPromotion,dataset.Attrition,dataset.NumCompaniesWorked)
print(stats,p)

# Mann whitney Test
# Ho = There is no significant difference NumCompaniesWorked and Attrition and Years Since Last Promotion
# Ha = There is  significant difference NumCompaniesWorked and Attrition and Years Since Last Promotion

from scipy.stats import mannwhitneyu
stats,p=mannwhitneyu(dataset.Attrition,dataset.JobLevel )
print(stats,p)

