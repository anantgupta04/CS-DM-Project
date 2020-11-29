# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 18:58:12 2020

@author: shubh
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from string import ascii_uppercase
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import joblib

scaler = MinMaxScaler()

# Read the input csv file
nutri_dataset = pd.read_csv("new_data.csv")

# Drop the product and nutriscore
nutri_dataset = nutri_dataset.drop(["Unnamed: 0","productname","nutriscorescore"], axis=1)

# Split the data into features and target
features = nutri_dataset.drop("nutriscoregrade", axis=1)
targets = nutri_dataset["nutriscoregrade"]

features[features.columns] = scaler.fit_transform(features[features.columns])

print(features)

Decision_tree_score= []
Random_forest_score= []
SVM_score = []
#Xgboost_score = []
# Split the data into a training and a testing set
#kf = KFold(n_splits = 3)

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, train_size=0.80)
#train_features, test_features, train_targets, test_targets = kf.split(features, targets)
#print(train_features,test_features)
# Split the data into KFold
kf = StratifiedKFold(n_splits=100)
for train_index, test_index in kf.split(features,targets):
    #print(train_index,test_index)
    train_features, test_features, train_targets, test_targets=features.iloc[train_index],features.iloc[test_index],targets.iloc[train_index],targets.iloc[test_index]
    #tree = DecisionTreeClassifier()
    tree = DecisionTreeClassifier(criterion='entropy',max_depth=None,min_samples_split=2)
    tree = tree.fit(train_features, train_targets)
    # Predict the new data
    DT_prediction = tree.predict(test_features)
    # Check the accuracy
    score = tree.score(test_features, test_targets)
    Decision_tree_score.append(score*100)
    
    # Train the model on RandomForestClassifier
    #model = RandomForestClassifier()
    model = RandomForestClassifier(n_estimators=100, max_features='sqrt', min_samples_leaf=1)
    model = model.fit(train_features, train_targets)

    # Predict the new data
    RD_prediction = model.predict(test_features)

    # Check the accuracy
    score = model.score(test_features, test_targets)
    Random_forest_score.append(score*100)
    
    # Train the model on SVM
    svm = SVC()    
    svm= svm.fit(train_features, train_targets)

    # Predict the new data
    SVM_prediction = model.predict(test_features)

    # Check the accuracy
    score = svm.score(test_features, test_targets)
    SVM_score.append(score*100)
    
    # Train the model on xgboost
    #model = RandomForestClassifier(n_estimators=100, max_features='sqrt', min_samples_leaf=1)
    #xgb_model = XGBClassifier()

    #xgb_model.fit(train_features, train_targets)

    
    # Predict the new data
    #prediction = xgb_model.predict(test_features)

    # Check the accuracy
    #score = xgb_model.score(test_features, test_targets)
    #Xgboost_score.append(score*100)
 
#print(Decision_tree_score)
Average_accuracy= sum(Decision_tree_score)/len(Decision_tree_score)
print('Decision Tree =',Average_accuracy)

#print(Random_forest_score)
Average_accuracy= sum(Random_forest_score)/len(Random_forest_score)
print('Random Forest =',Average_accuracy)

#print(SVM_score)
Average_accuracy= sum(SVM_score)/len(SVM_score)
print('SVM =',Average_accuracy)

# Confusion Matrix
y_actual = test_targets
y_prediction = DT_prediction
df_confusion = confusion_matrix(y_actual, y_prediction)
print('accuracy =', accuracy_score(y_actual,y_prediction))
    
columns = [i for i in list(ascii_uppercase)[0:len(np.unique(y_actual))]]    
df_cm = pd.DataFrame(df_confusion, index=columns, columns=columns)

ax = sn.heatmap(df_cm, cmap='Blues', annot=True)
plt.title("Desicion Tree Accuracy matrix")

#print(Xgboost_score)
#Average_accuracy= sum(Xgboost_score)/len(Xgboost_score)
#print(Average_accuracy)
#print("The prediction accuracy is: {:0.2f}%".format(Decision_tree_score * 100))
