#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pulp import (LpMaximize, LpVariable, lpSum, LpStatus, LpProblem)
import pandas as pd
import numpy as np

PATH = "C:\\Users\\akhilg\\Documents\\CollegeDocuments\\BDMA\\CentralSuperlec\\Coursework\\DM\\Assignments\\Final Project\\"
# PATH = "https://raw.github.com/anantgupta04/CS-DM-Project/main/"


def additive():

    def sample(data):
        total_len = len(data)
        cols = data.columns
        train_cols = cols.copy()
        x, y = data[cols].drop(['nutriscoregrade'],axis=1), data['nutriscoregrade']
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42)
        train = X_train.join(y_train)
        test = X_test.join(y_test)
        print("Test database  = \n\n", test)
        return train, y_test

    main_df = pd.read_excel(PATH + "OpenFood_Petales.xlsx")
    dataset = main_df.copy().sort_values(by=['nutriscorescore'])
    # dataset.reset_index(inplace=True)
    # train, test = sample(main_df)

    nutrigrades = dataset['nutriscoregrade'].unique()
    print("nutrigrades present in the DB are = ", nutrigrades)

    total_samples = len(dataset)
    feature_cols = ['energy100g','saturatedfat100g', 'sugars100g', 'fiber100g',
                'proteins100g',	'sodium100g']

    U_x = []
    Sigma = []
    U = pd.DataFrame(index=np.arange(total_samples),columns=feature_cols)

    # define the problem and eps
    prob = LpProblem("NutriScore", LpMaximize)
    eps = [LpVariable("epsilion_{}_{}".format(nutrigrades[i],nutrigrades[i+1]),0.1,10)
                for i in range(len(nutrigrades)-1)]
    print("Eps = ",eps)

    # Objective function
    prob += lpSum(eps)

    print("Length of the original in dataset = ",len(dataset.productname.values))

    # Pb. variables and utilities functions
    for ix,tup in dataset.iterrows():
        U_x += [LpVariable("U_{}".format(ix), 0, 10)] # problem variables
        # V_x += [LpVariable("V_{}".format(ix), 0, 20)]

        for crit in feature_cols:
            # print("The value for product {} has {} = {}. Type is {}".format(tup['productname'],
            #             crit, tup[crit], type(tup[crit])))
            U[crit][ix] = LpVariable('utility_{}_{}_{}'.format(crit, tup[crit],ix),0,1) #utility fn.



    # Contraints associated to the global utility of each food
    for ix,name in enumerate(dataset.productname):
        #print("Product Name = {1}\nUtility func'n = {0}\n ".format(U.loc[ix].values,name))
        prob += lpSum(U.loc[ix].values) == U_x[ix], 'cerealU_{1}_{0} contraint'.format(name,ix)

    maximize = ['fiber100g','proteins100g']
    for crit in feature_cols:
        if crit in maximize:
            sorted_c = dataset[crit].sort_values(ascending=True)
        else:
            sorted_c = dataset[crit].sort_values(ascending=False)
        for i in range(len(sorted_c)-1):
            try:
                index_1 = sorted_c.index[i]
                index_2 = sorted_c.index[i+1]
                if sorted_c[index_1] != sorted_c[index_2]:
                        prob += (U[crit][index_1] ) <= U[crit][index_2]
                elif sorted_c[index_1] == sorted_c[index_2]:
                    #print("inside elif")
                    prob += U[crit][index_1] == U[crit][index_2]
            except Exception as e:
                print("Exception occured = ",e)
                print("For crit = {} & i = {} & index_1 = {} & index 2 = {}".format(crit,i,sorted_c.index[i],i+1))
                assert False

    for ix in range(len(nutrigrades)-1):
        score_round = nutrigrades[ix]
        eps_round = eps[ix]
        g1 = dataset[dataset.nutriscoregrade == score_round].index
        g2 = dataset[dataset.nutriscoregrade == nutrigrades[ix+1]].index

        from itertools import cycle
        zip_list = zip(g1, cycle(g2)) if len(g1) > len(g2) else zip(cycle(g1), g2)
        #for i in subset[subset.nutriscoregrade == scores[ix+1]].index: #(subset.loc[subset.nutriscoregrade != ix]):
        for g1_i, g2_i in zip_list:
            print("\nInside print\n",dataset['nutriscoregrade'].iloc[g1_i],"------\t----",dataset['nutriscoregrade'].iloc[g2_i])
            print("Values of U_x[{}] is {}\tValues of U_x[{}] is {} ".format(g2_i,U_x[g2_i],g1_i,U_x[g1_i]))
            prob += (U_x[g2_i] + eps_round) <= U_x[g1_i]



    # The problem data is written to an .lp file
    prob.writeLP("The Nutriscore.lp")

    # solve model
    prob.solve()

    # The status of the solution is printed to the screen
    print("\n\n!!!Status: {}\n\n".format(LpStatus[prob.status]))

    # Each of the variables is printed with it's resolved optimum value
    import pdb; pdb.set_trace()
    utility_val = {}
    for i,v in enumerate(prob.variables()):

        if LpVariable(v.name) in U_x:
            try:
                number = int((v.name.split('_')[1]))
                utility_val[(number)] = v.varValue
            except Exception as e:
                continue
    print("Utility value is = ", utility_val)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    additive()
