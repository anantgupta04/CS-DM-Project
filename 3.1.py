#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:51:47 2020

@author: Cata
"""

from pulp import *
import pandas as pd
import numpy as np

dataset = pd.read_excel("OpenFood_Petales.xlsx")
subset = pd.read_excel("OpenFood_Petales.xlsx", sheetname="SubDataSet")
subset.sort_values('nutriscorescore',inplace=True)
subset.reset_index(inplace=True)
n_subset = len(subset)
L_crit = subset.columns[-6:]

U_x = []
V_x = []
Sigma = []
eps = 0.1
U = pd.DataFrame(index=np.arange(n_subset),columns=L_crit)

prob = LpProblem("NutriScore", LpMinimize)


# Normalize each column [0,1]
for crit in subset.columns[-6:]:
    subset[crit] = (subset[crit] - np.min(subset[crit]))/(np.max(subset[crit]) - np.min(subset[crit]))

# Pb. variables and utilities functions
for ix,name in enumerate(subset.productname):
    Sigma += [LpVariable('sigma_{}'.format(ix),0,1)]
    U_x += [LpVariable(name,0,6)] # problem variables
    V_x += [LpVariable(name,0,6)]
    for crit in subset.columns[-6:]:
        U[crit][ix] = LpVariable('utility_{}_{}'.format(crit, name),0,1) #utility fn.
     
# Objective function
for i in range(len(U_x)):
    prob += Sigma[i], "Adding all the sigma which we need to minimize."
    #prob += lpSum(Sigma)

# Contraints associated to the global utility of each food
maximize = ['fiber100g','proteins100g']
for ix,name in enumerate(subset.productname):    
    #import pdb;pdb.set_trace()
#    prob += lpSum(i for i in U.loc[ix]) == U_x[ix], 'cereal_{} contraint'.format(name)
    prob += lpSum(U.loc[ix]) == U_x[ix], 'cereal_{} contraint'.format(name)
    prob +=  U_x[ix] + Sigma[ix] == V_x[ix]
    print("u_x = ", U_x[ix])
    
for crit in L_crit:
    if crit in maximize:
        sorted_c = subset[crit].sort_values(ascending=True)
    else:
        sorted_c = subset[crit].sort_values(ascending=False)
    for ix in sorted_c.index:
        try:
            prob += U[crit][ix] <= U[crit][ix+1]
        except KeyError:
            continue
'''
scores = ['a','b','c','d','e']
for ix in range(len(scores)-1):
    score_round = scores[ix]
    tup = subset[subset.nutriscoregrade == score_round].index
    print("\n\ntup ={0}. Scores of this round= {1} ".format(tup,score_round))
    next_score = subset[subset.nutriscoregrade == scores[ix+1]].index
    len_max = max(len(tup),len(next_score))
    #for i in subset[subset.nutriscoregrade == scores[ix+1]].index: #(subset.loc[subset.nutriscoregrade != ix]):
    for i in range(len_max):
        great_one = tup[i]
        smaller_one = next_score[i]
        print("\nInside print\n",U_x[great_one],"------\t----",U_x[smaller_one])
        prob += (U_x[smaller_one]+eps) <= U_x[great_one]
 ''' 
 
#prob += 0<eps
# solve model
prob.solve()

# The status of the solution is printed to the screen
print("Statut:", LpStatus[prob.status])
# Output= 
# Status: Optimal

# Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    print(v.name, "=", v.varValue)