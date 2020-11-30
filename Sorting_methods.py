#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:51:47 2020

@author: Cata 
"""
#PATH = "https://raw.github.com/anantgupta04/CS-DM-Project/main/"

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
from string import ascii_uppercase
from Limiting_Profiles import limiting_profiles_dist
from additive import PATH


def PessimisticmajoritySorting(subset, pi, w, threshold):
    w_sum = sum(w.values())
    maximize = ['fiber100g','proteins100g']
    New_Subset = subset.copy()
    for threshold in [0.5, 0.6, 0.7]:
        New_Subset['pessimistic_grade_'+str(threshold)] = ""
    for index, tup in subset.iterrows():
        for p_index, value in (pi.iterrows()):
            s = 0.
            for crit in subset.columns[-6:]:
                if crit in maximize:
                    if tup[crit] >= value[crit]:
                        s += w[crit]
                else:
                    if tup[crit] <= value[crit]:
                        s += w[crit]
            diff = s / w_sum
            for threshold in [0.5, 0.6, 0.7]:
                if not New_Subset.loc[index,'pessimistic_grade_'+str(threshold)]\
                    and diff >= threshold:
                    if p_index in [0,1]:
                        grade = 'a'
                    elif p_index == 2:
                        grade ='b'
                    elif p_index == 3:
                        grade = 'c'
                    elif p_index == 4:
                        grade ='d'
                    elif p_index == 5:
                        grade = 'e'
                    New_Subset.loc[index,'pessimistic_grade_'+str(threshold)] = grade
                    continue

    y_actu = subset['nutriscoregrade']
    y_pred_05 = New_Subset['pessimistic_grade_0.5']
    y_pred_06 = New_Subset['pessimistic_grade_0.6']
    y_pred_07 = New_Subset['pessimistic_grade_0.7']
    df_confusion_05 = confusion_matrix(y_actu, y_pred_05)
    df_confusion_06 = confusion_matrix(y_actu, y_pred_06)
    df_confusion_07 = confusion_matrix(y_actu, y_pred_07)
    print('accuracy 0.5 =', accuracy_score(y_actu,y_pred_05))
    print('accuracy 0.6 =', accuracy_score(y_actu,y_pred_06))
    print('accuracy 0.7 =', accuracy_score(y_actu,y_pred_07))
    
    columns = [i for i in list(ascii_uppercase)[0:len(np.unique(y_actu))]]    
    df_cm_05 = pd.DataFrame(df_confusion_05, index=columns, columns=columns)
    df_cm_06 = pd.DataFrame(df_confusion_06, index=columns, columns=columns)
    df_cm_07 = pd.DataFrame(df_confusion_07, index=columns, columns=columns)
    ax_05 = sn.heatmap(df_cm_05, cmap='Reds', annot=True)
#    ax_06 = sn.heatmap(df_cm_06, cmap='Blues', annot=True)
#    ax_07 = sn.heatmap(df_cm_07, cmap='Greens', annot=True)
    plt.title("Pessimistic confusion matrix for lambda 0.5")
#    plt.title("Pessimistic confusion matrix for lambda 0.6")
#    plt.title("Pessimistic confusion matrix for lambda 0.7")
    


def OptimisticmajoritySorting(subset, pi, w, threshold_range):
    w_sum = sum(w.values())
    maximize = ['fiber100g','proteins100g']
    New_Op_Subset = subset.copy()

    pi = pi[::-1]  #reverse the dataframe
    pi.reset_index(inplace=True)

    thresh_check = {}
    for threshold in threshold_range:
        New_Op_Subset['optimistic_grade_'+str(threshold)] = None
        thresh_check[threshold] = True
    for index, tup in subset.iterrows():
        prod_threshold = thresh_check.copy()
        for p_index, value in pi.iterrows():
            pi_sum, s_sum, s_dash = 0., 0., 0
            if not any(prod_threshold.values()):
                break
            for crit in subset.columns[-6:]:
                if crit in maximize: # for criteria that we need to maximize
                    if tup[crit] < value[crit]:
                        pi_sum += w[crit]
                    elif tup[crit] == value[crit]:
                        pi_sum += w[crit]
                        s_sum += w[crit]
                    else:
                        s_sum += w[crit]
                else: # for criteria that we need to minimize
                    if tup[crit] > value[crit]:
                        pi_sum += w[crit]
                    elif tup[crit] == value[crit]:
                        pi_sum += w[crit]
                        s_sum += w[crit]
                    else:
                        s_sum += w[crit]

            diff = (s_sum) / w_sum
            s_dash = (pi_sum) / w_sum
            for threshold in threshold_range:
                grade = ""
                if (s_dash >= threshold) and (not diff >= threshold ) \
                    and not New_Op_Subset.loc[index,'optimistic_grade_'+str(threshold)]:
                    if value.pi.lower() in ["pi1", "pi2"]:
                        grade = 'e'
                    elif value.pi.lower() == "pi3":
                        grade = 'd'
                    elif value.pi.lower() == "pi4":
                        grade = 'c'
                    elif value.pi.lower() == "pi5":
                        grade = 'b'
                    elif value.pi.lower() == "pi6":
                        grade = 'a'
                    New_Op_Subset.loc[index,'optimistic_grade_'+str(threshold)] = grade
                    prod_threshold[threshold] = False

    # Confusion Matrix
    y_actu = subset['nutriscoregrade']
    y_pred_05 = New_Op_Subset['optimistic_grade_0.5']
    y_pred_06 = New_Op_Subset['optimistic_grade_0.6']
    y_pred_07 = New_Op_Subset['optimistic_grade_0.7']
    
    df_confusion_05 = confusion_matrix(y_actu, y_pred_05)
    df_confusion_06 = confusion_matrix(y_actu, y_pred_06)
    df_confusion_07 = confusion_matrix(y_actu, y_pred_07)
    print('accuracy = 0.5 ', accuracy_score(y_actu, y_pred_05))
    print('accuracy = 0.6 ', accuracy_score(y_actu, y_pred_06))
    print('accuracy = 0.7 ', accuracy_score(y_actu, y_pred_07))
#    
    columns = [i for i in list(ascii_uppercase)[0:len(np.unique(y_actu))]]    
    df_cm_05 = pd.DataFrame(df_confusion_05, index=columns, columns=columns)
    df_cm_06 = pd.DataFrame(df_confusion_06, index=columns, columns=columns)
    df_cm_07 = pd.DataFrame(df_confusion_07, index=columns, columns=columns)
    ax_05 = sn.heatmap(df_cm_05, cmap='Reds', annot=True)
#    ax_06 = sn.heatmap(df_cm_06, cmap='Blues', annot=True)
#    ax_07 = sn.heatmap(df_cm_07, cmap='Greens', annot=True)
    plt.title("Optimistic confusion matrix for lambda 0.5")
#    plt.title("Optimistic confusion matrix for lambda 0.6")
#    plt.title("Optimistic confusion matrix for lambda 0.7")




if __name__ == '__main__':
    threshold_range = [0.5,0.6,0.7]
    # data setup First Database
    dataset = pd.read_excel(PATH + "OpenFood_Petales.xlsx")
    dataset.sort_values('nutriscorescore',inplace=True)
    dataset.reset_index(inplace=True)
    w = {"energy100g":1,"saturatedfat100g":1,"sugars100g":1,"fiber100g":2,"proteins100g":2,"sodium100g":1}
    pi = pd.read_excel(PATH + "limiting_profiles.xlsx")
    
    # data setup Second Database
    dataset_2 = pd.read_csv(PATH + "new_data.csv")
    dataset_2.rename(columns={'Unnamed: 0':'index'}, inplace=True)
    titles = ['energy100g','saturatedfat100g','sugars100g','fiber100g','proteins100g','sodium100g']
    w2 = {"energy100g":4,"saturatedfat100g":1,"sugars100g":2,"fiber100g":6,"proteins100g":6,"sodium100g":1}
    pi2 = limiting_profiles_dist(dataset_2, titles)


    #Function calls for first database
#    PessimisticmajoritySorting(dataset, pi, w, [0.5,0.6,0.7])
#    OptimisticmajoritySorting(dataset, pi, w, threshold_range)
    
#Function calls for second database
    PessimisticmajoritySorting(dataset_2, pi2, w2, [0.5,0.6,0.7])
#    OptimisticmajoritySorting(dataset_2, pi2, w2, threshold_range)
