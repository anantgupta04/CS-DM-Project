# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 18:58:12 2020

@author: shubh
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
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

Decision_tree_score= []
Random_forest_score= []
SVM_score = []
# Split the data into a training and a testing set
#kf = KFold(n_splits = 3)

#for train_features, test_features in kf.split(features):
#    train_features, test_features, train_targets, test_targets = features(train_features),features(train_targets)\
                                                                 #features(test_features),features(test_targets)
 #   print(train_features,test_features)

#def get_score(model,train_features, test_features, train_targets, test_targets):
#    model.fit(train_features,train_targets)
#    return model.score(test_features,test_targets)


#train_features, test_features, train_targets, test_targets = train_test_split(features, targets, train_size=0.80)

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, train_size=0.80)
#train_features, test_features, train_targets, test_targets = kf.split(features, targets)
#print(train_features,test_features)
# Split the data into KFold
kf = StratifiedKFold(n_splits=10)
for train_index, test_index in kf.split(features,targets):
    #print(train_index,test_index)
    train_features, test_features, train_targets, test_targets=features.iloc[train_index],features.iloc[test_index],targets.iloc[train_index],targets.iloc[test_index]
    tree = DecisionTreeClassifier()
    #tree = DecisionTreeClassifier(criterion='entropy',max_depth=None,min_samples_split=2)
    tree = tree.fit(train_features, train_targets)
    # Predict the new data
    prediction = tree.predict(test_features)
    # Check the accuracy
    score = tree.score(test_features, test_targets)
    Decision_tree_score.append(score*100)
    
    # Train the model on RandomForestClassifier
    model = RandomForestClassifier()
    #model = RandomForestClassifier(n_estimators=100, max_features='sqrt', min_samples_leaf=1)
    model = model.fit(train_features, train_targets)

    # Predict the new data
    prediction = model.predict(test_features)

    # Check the accuracy
    score = model.score(test_features, test_targets)
    Random_forest_score.append(score*100)
    
    # Train the model on SVM
    svm = SVC()
    #model = RandomForestClassifier(n_estimators=100, max_features='sqrt', min_samples_leaf=1)
    svm= svm.fit(train_features, train_targets)

    # Predict the new data
    prediction = model.predict(test_features)

    # Check the accuracy
    score = svm.score(test_features, test_targets)
    SVM_score.append(score*100)
 
print(Decision_tree_score)
Average_accuracy= sum(Decision_tree_score)/len(Decision_tree_score)
print(Average_accuracy)

print(Random_forest_score)
Average_accuracy= sum(Random_forest_score)/len(Random_forest_score)
print(Average_accuracy)

print(SVM_score)
Average_accuracy= sum(SVM_score)/len(SVM_score)
print(Average_accuracy)
#print("The prediction accuracy is: {:0.2f}%".format(Decision_tree_score * 100))
'''
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
'''