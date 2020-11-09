#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:51:47 2020

@author: Cata lol
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
eps = 0.3
U = pd.DataFrame(index=np.arange(n_subset),columns=L_crit)

prob = LpProblem("NutriScore", LpMinimize)


# Normalize each column [0,1]
for crit in subset.columns[-6:]:
    subset[crit] = (subset[crit] - np.min(subset[crit]))/(np.max(subset[crit]) - np.min(subset[crit]))

# Pb. variables and utilities functions
for ix,name in enumerate(subset.productname.values):
    Sigma += [LpVariable('sigma_{}'.format(ix),0,1)]
    U_x += [LpVariable(name + '_u',0,6)] # problem variables
    V_x += [LpVariable(name + '_v',0,6)]
    for crit in subset.columns[-6:]:
        U[crit][ix] = LpVariable('utility_{}_{}'.format(crit, name),0,1) #utility fn.
     
# Objective function
prob += lpSum(Sigma)

# Contraints associated to the global utility of each food
maximize = ['fiber100g','proteins100g']
for ix,name in enumerate(subset.productname):
    prob += lpSum(U.loc[ix].values) == U_x[ix], 'cereal_{} contraint'.format(name)
    prob +=  U_x[ix] + Sigma[ix] == V_x[ix]
    
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
