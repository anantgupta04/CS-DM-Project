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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from string import ascii_uppercase
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np


PATH_1 = "OpenFood_Petales.xlsx"
PATH_2 = "new_data.csv"

def Plot_Matrix(matrix, y_actual, model_name):
    columns = [i for i in list(ascii_uppercase)[0:len(np.unique(y_actual))]]    
    df_cm = pd.DataFrame(matrix, index=columns, columns=columns)

    ax = sn.heatmap(df_cm, cmap='Blues', annot=True)
    plt.title(str(model_name) + " Accuracy matrix")

def ML_prediction(path):

    scaler = MinMaxScaler()
    
    if path == PATH_1:
        
        # Read the input csv file
        nutri_dataset = pd.read_excel(PATH_1)

        # Drop the product and nutriscore
        nutri_dataset = nutri_dataset.drop(["productname","nutriscorescore"], axis=1)
        
        # Split the data into KFold
        kf = StratifiedKFold(n_splits=10)
    
    elif path == PATH_2:
        # Read the input csv file
        nutri_dataset = pd.read_csv(PATH_2)

        # Drop the product and nutriscore
        nutri_dataset = nutri_dataset.drop(["Unnamed: 0","productname","nutriscorescore"], axis=1)
        
        # Split the data into KFold
        kf = StratifiedKFold(n_splits=100)
    

    # Split the data into features and target
    features = nutri_dataset.drop("nutriscoregrade", axis=1)
    targets = nutri_dataset["nutriscoregrade"]

    features[features.columns] = scaler.fit_transform(features[features.columns])

    print(features)

    Decision_tree_score= []
    Random_forest_score= []
    SVM_score = []
    final_conf_mat_DT = np.zeros([5,5])
    final_conf_mat_RF = np.zeros([5,5])
    final_conf_mat_SVM = np.zeros([5,5])

    train_features, test_features, train_targets, test_targets = train_test_split(features, targets, train_size=0.80)
    y_actual_org= test_targets.copy()
    for train_index, test_index in kf.split(features,targets):
        
        train_features, test_features, train_targets, test_targets=features.iloc[train_index],features.iloc[test_index],targets.iloc[train_index],targets.iloc[test_index]
        
        #tree = DecisionTreeClassifier()
        tree = DecisionTreeClassifier(criterion='entropy',max_depth=None,min_samples_split=2)
        tree = tree.fit(train_features, train_targets)
        # Predict the new data
        DT_prediction = tree.predict(test_features)
        
        # Confusion Matrix
        DT_cf = confusion_matrix(test_targets, DT_prediction)
        final_conf_mat_DT = np.add(final_conf_mat_DT,DT_cf)
        
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
        
        # Confusion Matrix
        RF_cf = confusion_matrix(test_targets, RD_prediction)
        final_conf_mat_RF = np.add(final_conf_mat_RF,RF_cf)
        
        
        # Train the model on SVM
        svm = SVC(kernel='rbf', class_weight="balanced")    
        svm= svm.fit(train_features, train_targets)
        
        # Predict the new data
        SVM_prediction = svm.predict(test_features)
        
        # Confusion Matrix
        SVM_cf = confusion_matrix(test_targets, SVM_prediction)
        #print("Type of confusion matrix = ", type(c1))
        final_conf_mat_SVM = np.add(final_conf_mat_SVM,SVM_cf)
        #print("New confusion matrix is = ", final_conf_mat)
        
        # Check the accuracy
        score = svm.score(test_features, test_targets)
        SVM_score.append(score*100)
        
    
    #plot Decision_tree
    Average_accuracy= sum(Decision_tree_score)/len(Decision_tree_score)
    print('Decision Tree =',Average_accuracy)
    
    #y_actual = test_targets
    #y_prediction = DT_prediction  
    #df_confusion_DT = confusion_matrix(y_actual, y_prediction)
    #print('accuracy =', accuracy_score(y_actual,y_prediction))
    model_name = 'Decision Tree'
    
    plot = Plot_Matrix(final_conf_mat_DT,y_actual_org,model_name)

    #plot Random_forest
    Average_accuracy= sum(Random_forest_score)/len(Random_forest_score)
    print('Random Forest =',Average_accuracy)
    
    model_name= 'Random Forest'
    
    plot = Plot_Matrix(final_conf_mat_RF,y_actual_org,model_name)

      
    #plot SVM
    Average_accuracy= sum(SVM_score)/len(SVM_score)
    print('SVM =',Average_accuracy)

    model_name = 'SVM'
    
    #plot = Plot_Matrix(final_conf_mat_SVM,y_actual_org,model_name)

if __name__ == "__main__":
    Open_petal = ML_prediction(PATH_1) 
    Our_data = ML_prediction(PATH_2)
