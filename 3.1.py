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
#for v in prob.variables():
#    print(v.name, "=", v.varValue)
