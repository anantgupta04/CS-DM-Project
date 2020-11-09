#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:51:47 2020

@author: Cata lol
"""

# PATH = "https://github.com/anantgupta04/CS-DM-Project/blob/main/OpenFood_Petales.xlsx?raw=true"
PATH = "C:\\Users\\akhilg\\Documents\\CollegeDocuments\\BDMA\\CentralSuperlec\\Coursework\\DM\\Assignments\\Final Project\\"
# PATH = "https://raw.github.com/anantgupta04/CS-DM-Project/main/"



from pulp import (LpMinimize, LpVariable, lpSum, LpStatus, LpProblem)
import pandas as pd
import numpy as np

# reading the data 1
dataset = pd.read_excel(PATH + "OpenFood_Petales.xlsx")
subset = pd.read_excel((PATH + "OpenFood_Petales.xlsx"), sheet_name="SubDataSet")

subset.sort_values('nutriscorescore',inplace=True)
subset.reset_index(inplace=True)
n_subset = len(dataset)
L_crit = subset.columns[-6:]

U_x = []
V_x = []
Sigma = []
<<<<<<< HEAD
=======
eps = 0.3
>>>>>>> b649aa64ccb7abcee47329f0e7e69d9ccef9e401
U = pd.DataFrame(index=np.arange(n_subset),columns=L_crit)

# define the problem and eps
prob = LpProblem("NutriScore", LpMinimize)
eps  = LpVariable("Eps",0,None)

# Normalize each column [0,1]
'''
for crit in dataset.columns[-6:]:
    dataset[crit] = (dataset[crit] - np.min(dataset[crit]))/(np.max(dataset[crit]) - np.min(dataset[crit]))
'''

print("Length of the original in dataset = ",len(dataset.productname.values))

# Pb. variables and utilities functions
for ix,tup in dataset.iterrows():
    Sigma += [LpVariable("sigma_{}".format(ix), 0, 10)] # list of all Sigma's
    # print("\nSigma = {}\n".format(Sigma))

    U_x += [LpVariable("U_{}".format(ix), 0, 10)] # problem variables
    V_x += [LpVariable("V_{}".format(ix), 0, 20)]

    # print("U_x for {0} is {1}.".format(tup['productname'],U_x))
    # print("V_x for {0} is {1}.".format(tup['productname'],V_x))


    for crit in L_crit:
        # print("The value for product {} has {} = {}. Type is {}".format(tup['productname'],
        #             crit, tup[crit], type(tup[crit])))

        U[crit][ix] = LpVariable('utility_{}_{}_{}'.format(crit, tup[crit],ix),0,1) #utility fn.

    # print("\n\n",U.iloc[ix])


# Objective function
prob += lpSum(Sigma)

# Contraints associated to the global utility of each food
for ix,name in enumerate(dataset.productname):
    #print("Product Name = {1}\nUtility func'n = {0}\n ".format(U.loc[ix].values,name))
    prob += lpSum(U.loc[ix].values) == U_x[ix], 'cerealU_{1}_{0} contraint'.format(name,ix)
    prob +=  U_x[ix] + Sigma[ix] == V_x[ix], 'cerealV_{1}_{0} contraint'.format(name,ix)
    # print("Problem Space is = ", prob)
    # assert False

maximize = ['fiber100g','proteins100g']
for ix,crit in enumerate(L_crit,1):
    eps1  = LpVariable("eps_{}".format(str(ix)), 0.1, 10)
    if crit in maximize:
        sorted_c = dataset[crit].sort_values(ascending=True)
    else:
        sorted_c = dataset[crit].sort_values(ascending=False)
    for i in range(len(sorted_c)-1):
        try:
            index_1 = sorted_c.index[i]
            index_2 = sorted_c.index[i+1]
            if sorted_c[i] != sorted_c[i+1]:
                    prob += (U[crit][index_1] + eps1) <= U[crit][index_2]
            elif sorted_c[i] == sorted_c[i+1]:
                print("inside elif")
                prob += U[crit][index_1] == U[crit][index_2]
        except KeyError:
            continue

scores = ['a','b','c','d','e']
for ix in range(len(scores)-1):
    score_round = scores[ix]
    tup = subset[subset.nutriscoregrade == score_round].index
    #print("\n\ntup ={0}. Scores of this round= {1} ".format(tup,score_round))
    next_score = subset[subset.nutriscoregrade == scores[ix+1]].index
    len_max = max(len(tup),len(next_score))
    #for i in subset[subset.nutriscoregrade == scores[ix+1]].index: #(subset.loc[subset.nutriscoregrade != ix]):
    for i in range(len_max):
        great_one = tup[i]
        smaller_one = next_score[i]
        #print("\nInside print\n",U_x[great_one],"------\t----",U_x[smaller_one])
        prob += (U_x[smaller_one] + eps) <= U_x[great_one]

import pdb; pdb.set_trace()
# eps_string = [ 'eps_{}'.format(i) for i in range(1,len(dataset)+1)]
# print("Created epsilion string = \n", eps_string)
#
# prob += 0.1<eps, "Rank preference epsilion"
# prob += [(0.1<eps, "Rank preference for {}".format(eps)) for eps in eps_string ]
# solve model
prob.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])
# Output= # Status: Optimal

# Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    print(v.name, "=", v.varValue)


#%%

# weight and limiting profiles for OpenFood_Petales
w = {"energy100g":1,"saturatedfat100g":1,"sugars100g":1,"fiber100g":2,"proteins100g":2,"sodium100g":1}
pi = pd.read_excel("limiting_profiles.xlsx")

def PessimisticmajoritySorting(subset, pi, w, threshold):
    w_sum = sum(w.values())
#    passpi = pd.DataFrame(index=subset.productname, columns=pi.pi)
    passpi = pd.DataFrame(index=subset.index, columns=pi.pi)
    maximize = ['fiber100g','proteins100g']
    New_Subset = subset.copy()
#    for threshold in [0.5, 0.6, 0.7]:
#        New_Subset['pessimistic_grade_'+str(threshold)] = ""
    for ix, name in enumerate(subset.productname):
        print('\nprod =', name)
        for p in pi.pi:
            s = 0
            print('\npi =',p)
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
    New_Subset.to_csv("New_Subsetito.csv", header=True)

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
                if crit not in maximize:
                    if tup[crit] <= value[crit]:
                        s += w[crit]
                        print('minimize =',crit)
                        print('s =',s)
                else:
                    if tup[crit] >= value[crit]:
                        s += w[crit]
                        print('maxi =',crit)
                        print('s =',s)
            diff = s / w_sum
            for threshold in [0.5, 0.6, 0.7]:
                if diff >= threshold and not New_Subset.loc[index,'pessimistic_grade_'+str(threshold)] :
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

    New_Subset.to_csv("New_Subseeet.csv", header=True)


def OptimisticmajoritySorting():
    pass









if __name__ == '__main__':
    dataset = pd.read_excel("OpenFood_Petales.xlsx")
    subset = pd.read_excel("OpenFood_Petales.xlsx", sheetname="SubDataSet",
                           headers=True)
    subset.sort_values('nutriscorescore',inplace=True)
    subset.reset_index(inplace=True)

    w = {"energy100g":1,"saturatedfat100g":1,"sugars100g":1,"fiber100g":2,"proteins100g":2,"sodium100g":1}
    pi = pd.read_excel("limiting_profiles.xlsx")

    df = subset.copy()
    Pess_1(df, pi, w, threshold)
