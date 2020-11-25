# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 18:58:12 2020

@author: shubh
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

scaler = MinMaxScaler()

# Read the input excel file
nutri_dataset = pd.read_excel("OpenFood_Petales.xlsx")

# Drop the product and nutriscore
nutri_dataset = nutri_dataset.drop(["productname","nutriscorescore"], axis=1)

# Split the data into features and target
features = nutri_dataset.drop("nutriscoregrade", axis=1)
targets = nutri_dataset["nutriscoregrade"]

features[features.columns] = scaler.fit_transform(features[features.columns])

print(features)

# Split the data into a training and a testing set
train_features, test_features, train_targets, test_targets = train_test_split(features, targets, train_size=0.80)

#train_features, test_features, train_targets, test_targets = train_test_split(features, targets, train_size=0.80)

        
# Train the model on DecisionTreeClassifier
#tree = DecisionTreeClassifier()
tree = DecisionTreeClassifier(criterion='entropy',max_depth=None,min_samples_split=2)
tree = tree.fit(train_features, train_targets)

# Predict the new data
prediction = tree.predict(test_features)

# Check the accuracy
score = tree.score(test_features, test_targets)
print("The prediction accuracy is: {:0.2f}%".format(score * 100))

# Train the model on RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_features='sqrt', min_samples_leaf=1)
model = model.fit(train_features, train_targets)

# Predict the new data
prediction = model.predict(test_features)

# Check the accuracy
score = model.score(test_features, test_targets)
print("The prediction accuracy is: {:0.2f}%".format(score * 100))
