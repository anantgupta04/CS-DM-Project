#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:51:47 2020

@author: Cata lol
"""

# PATH = "https://github.com/anantgupta04/CS-DM-Project/blob/main/OpenFood_Petales.xlsx?raw=true"
PATH = "C:\\Users\\akhilg\\Documents\\CollegeDocuments\\BDMA\\CentralSuperlec\\Coursework\\DM\\Assignments\\Final Project\\"
# PATH = "https://raw.github.com/anantgupta04/CS-DM-Project/main/"



from pulp import (LpMaximize, LpVariable, lpSum, LpStatus, LpProblem)
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
#%%

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def plot_conf_matrix(df_confusion, title='Confusion matrix for lambda =', cmap=plt.get_cmap('Reds')):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)

    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)




def PessimisticmajoritySorting(subset, pi, w, threshold):
    w_sum = sum(w.values())
#    passpi = pd.DataFrame(index=subset.productname, columns=pi.pi)
    passpi = pd.DataFrame(index=subset.index, columns=pi.pi)
    maximize = ['fiber100g','proteins100g']
    New_Subset = subset.copy()
#    for threshold in [0.5, 0.6, 0.7]:
#        New_Subset['pessimistic_grade_'+str(threshold)] = ""
    for ix, name in enumerate(subset.productname):
#        print('\nprod =', name)
        for p in pi.pi:
            s = 0
#            print('\npi =',p)
            for crit in subset.columns[-6:]:
                if crit not in maximize:
                    if subset[crit][ix] <= pi[pi.pi==p][crit].values:
                        s += w[crit]
                else:
                    if subset[crit][ix] >= pi[pi.pi==p][crit].values:
                        s += w[crit]

            if s/w_sum >= threshold:
#                passpi.loc[name,p] = True'
                passpi.loc[ix,p] = True
                break
            else:
#                passpi.loc[name,p] = False
                passpi.loc[ix,p] = False

    for ix, line in passpi.iterrows():
        for pi in line.index:
            if line[pi] == True:
                if pi == 'pi6' or pi == 'pi5':
                    grade = 'a'
                elif pi == 'pi4':
                    grade = 'b'
                elif pi == 'pi3':
                    grade = 'c'
                elif pi == 'pi2':
                    grade = 'd'
                else:
                    grade = 'e'
                print('grade =', grade)
                New_Subset.loc[ix,'pessimistic_grade_'+str(threshold)] = grade
            elif line[pi] == None:
                break
#        assert False
#    return passpi
#    New_Subset.to_csv("New_Subsetito.csv", header=True)
    y_actu = subset['nutriscoregrade']
    y_pred = New_Subset['pessimistic_grade_'+str(threshold)]
    print(confusion_matrix(y_actu, y_pred))
    df_confusion = pd.crosstab(y_actu, y_pred)
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    print(df_confusion)
    print(df_conf_norm)
    print('accuracy =', accuracy_score(y_actu,y_pred))
    plot_conf_matrix(df_confusion, title='Confusion matrix for lambda ='.format(threshold))

def Pess_1(subset, pi, w, threshold):
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
                        print('maximize = ',crit)
                        print('s =',s)
                else:
                    if tup[crit] <= value[crit]:
                        s += w[crit]
                        print('minimize = ',crit)
                        print('s =',s)
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

#    New_Subset.to_csv("New_Subseeet.csv", header=True)
    y_actu = subset['nutriscoregrade']
    y_pred_05 = New_Subset['pessimistic_grade_0.5']
    y_pred_06 = New_Subset['pessimistic_grade_0.6']
    y_pred_07 = New_Subset['pessimistic_grade_0.7']
#    print(confusion_matrix(y_actu, y_pred))
#    df_confusion = pd.crosstab(y_actu, y_pred)
#    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
#    print(df_confusion)
#    print(df_conf_norm)
#    print('accuracy 0.5 =', accuracy_score(y_actu,y_pred_05))
#    print('accuracy 0.6 =', accuracy_score(y_actu,y_pred_06))
#    print('accuracy 0.7 =', accuracy_score(y_actu,y_pred_07))
    plot_conf_matrix(df_confusion, title='Confusion matrix for lambda ='.format(threshold))


def Optimistic_Anant(subset, pi, w, threshold_range):
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
        # print("--"*30)
        # print('\nproduct = ', tup['productname'])
        prod_threshold = thresh_check.copy()
        for p_index, value in pi.iterrows():
            pi_sum, s_sum, s_dash = 0., 0., 0
            if not any(prod_threshold.values()):
                break
            # print('\npi =', value.pi)
            for crit in subset.columns[-6:]:
                # print("Criteria is = {} and tup-value is {}| P-table value is {}".format(crit, tup[crit], value[crit]))
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
            # print('Value of pi_sum = {} | pi_sum/w | s_sum = {} | s/w = {}'.format(pi_sum,s_dash,s_sum,diff))
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
                    # print("At threshold = {} | Grade = {} | Flag check = {}".format(threshold,grade,prod_threshold[threshold]))
        # print("\n\n")
    # New_Op_Subset.to_csv("New_Op_Anant_Test.csv", header=True)
    y_actu = subset['nutriscoregrade']
    y_pred_05 = New_Op_Subset['optimistic_grade_0.5']
    y_pred_06 = New_Op_Subset['optimistic_grade_0.6']
    y_pred_07 = New_Op_Subset['optimistic_grade_0.7']
#    print(confusion_matrix(y_actu, y_pred))
    df_confusion_05 = pd.crosstab(y_actu, y_pred_05)
    df_confusion_06 = pd.crosstab(y_actu, y_pred_06)
    df_confusion_07 = pd.crosstab(y_actu, y_pred_07)
#    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    print(df_confusion_07)
#    print(df_conf_norm)
    
    print('accuracy 0.5 =', accuracy_score(y_actu,y_pred_05))
    print('accuracy 0.6 =', accuracy_score(y_actu,y_pred_06))
    print('accuracy 0.7 =', accuracy_score(y_actu,y_pred_07))
    plot_conf_matrix(df_confusion_07, title='Confusion matrix for lambda ='.format(threshold))



def Optimistic_Cata(subset, pi, w, threshold):
    w_sum = sum(w.values())
    passpi = pd.DataFrame(index=subset.index, columns=pi.pi)
    maximize = ['fiber100g','proteins100g']
    New_Op_Subset = subset.copy()

    pi = pi[::-1]  #reverse the dataframe
    pi.reset_index(inplace=True)
    print(pi)
#    for threshold in [0.5, 0.6, 0.7]:
#        New_Subset['pessimistic_grade_'+str(threshold)] = ""
    for ix, name in enumerate(subset.productname):
        print('\nprod =', name)
        for p in pi.pi:
            s = 0
            
            print('\npi =',p)
            for crit in subset.columns[-6:]:
                if crit not in maximize:
                    if subset[crit][ix] >= pi[pi.pi==p][crit].values :
                        s += w[crit]
                        print('min =', crit)
                        print('s =', s)
                else:
                    if subset[crit][ix] <= pi[pi.pi==p][crit].values:
                        s += w[crit]
                        print('max =', crit)
                        print('s =', s)

            if s/w_sum > threshold: # CHANGES FROM >= to >
                passpi.loc[ix,p] = True
                break
            else:
                passpi.loc[ix,p] = False

    for ix, line in passpi.iterrows():
        for pi in line.index:
            if line[pi] == True:
                if pi == 'pi6':
                    grade = 'a'
                elif pi == 'pi5':
                    grade = 'b'
                elif pi == 'pi4':
                    grade = 'c'
                elif pi == 'pi3':
                    grade = 'd'
                elif pi in ['pi1', 'pi2']:
                    grade = 'e'
                print('grade =', grade)
                New_Op_Subset.loc[ix,'optimistic_grade_'+str(threshold)] = grade
            elif line[pi] == None:
                break
#        assert False

    print(passpi)
    New_Op_Subset.to_csv("New_Op_Cata.csv", header=True)
    y_actu = subset['nutriscoregrade']
    y_pred = New_Op_Subset['optimistic_grade_'+str(threshold)]
#    print(confusion_matrix(y_actu, y_pred))
    df_confusion = pd.crosstab(y_actu, y_pred)
#    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
#    print(df_confusion)
#    print(df_conf_norm)
    print('accuracy =', accuracy_score(y_actu,y_pred))
    plot_conf_matrix(df_confusion, title='Confusion matrix for lambda ='.format(threshold))





if __name__ == '__main__':
    threshold_range = [0.5,0.6,0.7]
    # data setup
    dataset = pd.read_excel(PATH + "OpenFood_Petales.xlsx")
    subset = pd.read_excel(PATH + "OpenFood_Petales.xlsx", sheet_name="SubDataSet",
                           headers=True)
    subset.sort_values('nutriscorescore',inplace=True)
    subset.reset_index(inplace=True)


    w = {"energy100g":1,"saturatedfat100g":1,"sugars100g":1,"fiber100g":2,"proteins100g":2,"sodium100g":1}
    # pi = pd.read_excel(PATH + "limiting_profiles.xlsx")

    df = subset.copy()


    # function calls
#    Pess_1(dataset, pi, w, [0.5,0.6,0.7])

#    PessimisticmajoritySorting(subset, pi, w, 0.5)
    # Optimistic_Anant(dataset, pi, w, threshold_range)\
    # Optimistic_Anant(dataset.iloc[40:44], pi, w, threshold_range)
#    Optimistic_Cata(dataset, pi, w, 0.7)
#