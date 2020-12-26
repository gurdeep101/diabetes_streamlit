#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 16:40:57 2020

@author: gurdeep

recode of https://medium.com/towards-artificial-intelligence/how-i-build-machine-learning-apps-in-hours-a1b1eaa642ed

"""
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

# load data

diab = pd.read_csv('../data/pima_indians_diabetes.csv')

# replace zeros with NaNs
diab[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diab[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

def impute_median(data, var):
    # function to impute the missing values with median based on outcome class
    temp = data[data[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median()
    data.loc[(data['Outcome'] == 0 ) & (data[var].isnull()), var] = temp.loc[0 ,var]
    data.loc[(data['Outcome'] == 1 ) & (data[var].isnull()), var] = temp.loc[1 ,var]
    return data

# impute values using the function
diab = impute_median(diab, 'Glucose')
diab = impute_median(diab, 'BloodPressure')
diab = impute_median(diab, 'SkinThickness')
diab = impute_median(diab, 'Insulin')
diab = impute_median(diab, 'BMI')

# separate features and target as x & y
y = diab['Outcome']
x = diab.drop('Outcome', axis = 1)
columns = x.columns

# scale values using StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(x)
X = scaler.transform(x)
features = pd.DataFrame(X, columns = columns)

# split into training and test set
xtrain, xtest, ytrain, ytest = train_test_split(features, y, test_size = 0.2, random_state = 42)

# define the model
model = RandomForestClassifier(n_estimators = 300, bootstrap = True, max_features = 'sqrt')
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

print(classification_report(ytest, ypred))
 
#inference input data
pregnancies = 2
glucose = 13
bloodpressure = 30
skinthickness = 4
insulin = 5
bmi = 5
dpf = 0.55
age = 34
feat_cols = features.columns

feat_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
row = [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]

#transform the inference data same as training data
df = pd.DataFrame([row], columns = feat_cols)
X = scaler.transform(df)
features = pd.DataFrame(X, columns = feat_cols)

dump(scaler, '../models/scaler.joblib')

#make predictions using the already built model [0: healthy, 1:diabetes]
if (model.predict(features)==0):
    print("This is a healthy person!")
else: print("This person has high chances of having diabetics!")














