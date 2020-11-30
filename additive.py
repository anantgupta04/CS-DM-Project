#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pulp import (LpMaximize, LpVariable, lpSum, LpStatus, LpProblem)
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

PATH = "C:\\Users\\akhilg\\Documents\\CollegeDocuments\\BDMA\\CentralSuperlec\\Coursework\\DM\\Assignments\\Final Project\\"
# PATH = "https://raw.github.com/anantgupta04/CS-DM-Project/main/"


FEATURES = ['energy100g','saturatedfat100g', 'sugars100g', 'fiber100g',
            'proteins100g',	'sodium100g']
MAXIMIZE = ['fiber100g','proteins100g']
GRADES = ["a", "b", "c", "d", "e"]


"""
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
    """

class Additive():

    def __init__(self,choice,initial=0,terminal=25):
        self.samples = 0
        self.initial = initial
        self.terminal = terminal
        self.choice = choice
        self.prob = LpProblem(f"NutriScore Additive Model_{str(choice)}",
                            LpMaximize)
        self.eps = []
        self.dataset, self.preferences = pd.DataFrame(), pd.DataFrame()

        self.U_x = [] # to store the global utility per food
        self.U = pd.DataFrame() # to store the marginal_utilities per food

    def execute(self):
        self._setup()
        self._objective_variables()
        self._global_marginal_setup()
        self._monotonicity_constraints()
        self._rank_preference()
        # The problem data is written to an .lp file
        # self.prob.writeLP(f"NutriScore Additive Model_{str(self.choice)}.lp")

        # solve model
        self.prob.solve()

        # The status of the solution is printed to the screen
        print(f"\n\n!!!Status: {LpStatus[self.prob.status]}\n\n")
        if self.prob.status:
            print("***"*20)
            print("\nCongratulations the problem has been successfully solved.\n")
            print("***"*20)
            self._save_results()
            self._change_path()
            self._plot_global_utility()
            self._plot_attributes()
        return self.dataset


    def _setup(self):
        if self.choice == 1: # OpenFood_Petales as choice 0
            main_df = pd.read_excel(PATH + "OpenFood_Petales.xlsx")
            self.dataset = main_df.copy().sort_values(by=['nutriscorescore'])
            self.dataset.reset_index(inplace=True, drop=True)
            self.preference = pd.read_excel(PATH + "\data\OpenFood_Petales_Preference.xlsx")
        else:  # Our own scrapped DB
            main_df, self.preference = split_second_db()
            self.dataset = main_df.copy().sort_values(by=['nutriscorescore'])
            self.dataset.reset_index(inplace=True, drop=True)
        # some checks on the length and the grades of the SubDataSet
        nutrigrades = self.dataset['nutriscoregrade'].unique()
        print("nutrigrades present in the DB are = ", nutrigrades)

        self.samples = len(self.dataset)

    def _objective_variables(self):
        # define the eps
        self.eps = [LpVariable(f"epsilion_{GRADES[i]}_{GRADES[i+1]}",
                        0,10) for i in range(len(GRADES)-1)]
        print("Eps = ", self.eps)

        # Objective function
        self.prob += lpSum(self.eps)

    def _global_marginal_setup(self):
        self.U = pd.DataFrame(index=np.arange(self.samples),columns=FEATURES)
        # Pb. variables and utilities functions
        for ix,tup in self.dataset.iterrows():
            varVariable = LpVariable(f"U_{ix}", self.initial, self.terminal)
            self.U_x += [varVariable] # problem variables
            for crit in FEATURES:
                k = f"utility_{crit}_{tup[crit]}_{ix}"
                self.U[crit][ix] = LpVariable(k,self.initial,self.terminal)

        # Contraints associated to the global utility of each food
        for idx,name in enumerate(self.dataset.productname):
            #print("Product Name = {1}\nUtility func'n = {0}\n ".format(U.loc[ix].values,name))
            self.prob += lpSum(self.U.loc[idx].values) == self.U_x[idx], \
                            f"cerealU_{idx}_{name} contraint"

    def _monotonicity_constraints(self):
        for crit in FEATURES:
            if crit in MAXIMIZE:
                sorted_c = self.dataset[crit].sort_values(ascending=True)
            else:
                sorted_c = self.dataset[crit].sort_values(ascending=False)
            for i in range(len(sorted_c)-1):
                try:
                    index_1 = sorted_c.index[i]
                    index_2 = sorted_c.index[i+1]
                    if sorted_c[index_1] != sorted_c[index_2]:
                            self.prob += (self.U[crit][index_1] ) <= \
                                self.U[crit][index_2]
                    elif sorted_c[index_1] == sorted_c[index_2]:
                        #print("inside elif")
                        self.prob += self.U[crit][index_1] == self.U[crit][index_2]
                except Exception as e:
                    print("Exception occured = ",e)
                    print("For crit = {} & i = {} & index_1 = {} & index 2 = {}".format(crit,i,sorted_c.index[i],i+1))
                    assert False

    def _rank_preference(self):
        for idx in range(len(GRADES)-1):
            score_round = GRADES[idx]
            eps_round = self.eps[idx]
            g1 = self.preference[self.preference.nutriscoregrade == score_round].index
            g2 = self.preference[self.preference.nutriscoregrade == GRADES[idx+1]].index
            # import pdb; pdb.set_trace()
            from itertools import cycle
            zip_list = zip(g1, cycle(g2)) if len(g1) > len(g2) else zip(cycle(g1), g2)
            for g1_i, g2_i in zip_list:
                try:
                    high = self.dataset.iloc[
                        np.where(self.dataset.productname ==
                            self.preference.iloc[g1_i].productname )
                        ].index[0]
                    low = self.dataset.iloc[
                        np.where(self.dataset.productname ==
                            self.preference.iloc[g2_i].productname )
                        ].index[0]
                except Exception as e:
                    print("Exception inside for for preference\nvalue of g2_i is {} and g1_i {}".format(g2_i,g1_i))
                    assert False
                # print("\nInside print\n",self.dataset['nutriscoregrade'].iloc[low],"------\t----",self.dataset['nutriscoregrade'].iloc[high])
                # print("Values of U_x[{}] is {}\tValues of U_x[{}] is {} ".format(low,self.U_x[low],high,self.U_x[high]))
                self.prob += (self.U_x[low] + eps_round) <= self.U_x[high]

    def _save_results(self):
        additive_scores = [i.varValue for i in self.U_x]
        self.dataset['additive_score'] = additive_scores

    def _plot_global_utility(self):
        f, ax = plt.subplots(figsize=(10, 8))
        sns.despine(f, left=True, bottom=True)
        sns_plot = sns.scatterplot(x="nutriscorescore", y="additive_score",
                                   hue="nutriscoregrade",palette="deep",
                                   data=self.dataset, ax=ax)
        sns_plot.get_figure().savefig(f"additive_score_inference.png")

    def _plot_attributes(self):

        for criterion in FEATURES:
            values_df = self.dataset[[criterion]]
            variable_values = [i.varValue for i in self.U[criterion]]

            values_df["marginal_utility_value"] = variable_values
            # kind="scatter",
            sns.relplot( x=criterion, y="marginal_utility_value",
                        data=values_df)
            plt.savefig(f"{criterion}_mariginal_utility_plot.png")

    '''
    dataset.to_csv("calculated scores.csv")
    '''

    def _change_path(self):
        choice = "OpenFood_Petales" if self.choice ==1 else "Own Database11"
        target = f"./Images/Additive/{choice}"
        if not os.path.exists(target):
            # print("\n\n GRADE-{} Folder created successfully".format(grade.upper()))
            os.makedirs(target, mode=0o777)
            os.chdir(target)

def split_second_db():
    # read the big DB

    main_df = pd.read_excel(PATH + "data\\additive_2.xlsx",
                        sheet_name="new_data")
    # preference = pd.read_excel(PATH + "data\\additive_2.xlsx",
    #                     sheet_name="Preference")

    main_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    print("length of db is = ", len(main_df))

    new_df = pd.DataFrame()
    preference = pd.DataFrame()
    for grade in GRADES:
        # sheet = f"Grade {grade.upper()}"
        grade_sheet = main_df.iloc[
                    np.where(main_df.nutriscoregrade == grade)
                ].sample(n=200, random_state=40)
        new_df = pd.concat([new_df, grade_sheet])
        preference_sheet = grade_sheet.sample(n=60, random_state=10)
        preference = pd.concat([preference, preference_sheet])

    new_df.to_csv("data\own_DB11.csv")
    preference.to_csv("data\own_DB_preference11.csv")
    preference.reset_index(inplace=True, drop=True)
    return new_df, preference



if __name__ == '__main__':
    # ob1 = Additive(1)
    # ob1.execute()


    ob2 = Additive(2)
    ob2.execute()
