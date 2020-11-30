#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:41:01 2020

@author: Cata
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv("data/new_data.csv")

titles = ['energy100g','saturatedfat100g','sugars100g','fiber100g','proteins100g','sodium100g']

def limiting_profiles(dataset, titles):
    limit_prof = pd.DataFrame(columns=titles)
    limit_prof.insert(0, 'pi', ['pi6','pi5','pi4','pi3','pi2','pi1'])
    maximize = ['fiber100g','proteins100g']
    limits = (0, 0.2, 0.4, 0.6, 0.8, 1)
    for name in titles:
        min_val = min(dataset[name])
        max_val = max(dataset[name])
        cat_len = max_val - min_val
        if name in maximize:
            for ix, value in enumerate(limits[::-1]):
                limit_prof[name][ix] = round(cat_len * value + min_val, 1)
        else:
            for ix, value in enumerate(limits):
                limit_prof[name][ix] = round(cat_len * value + min_val, 1)
    print(limit_prof)
    return limit_prof


def limiting_profiles_dist(dataset, titles):
    limit_prof = pd.DataFrame(columns=titles)
    limit_prof.insert(0, 'pi', ['pi6','pi5','pi4','pi3','pi2','pi1'])
    maximize = ['fiber100g','proteins100g']
    limits = (0, 20, 40, 60, 80, 100)
    for name in titles:
        if name in maximize:
            for ix, value in enumerate(limits[::-1]):
                limit_prof[name][ix] = round(np.percentile(dataset[name], value), 1)
        else:
            for ix, value in enumerate(limits):
                limit_prof[name][ix] = round(np.percentile(dataset[name], value), 1)
    print(limit_prof)
    return limit_prof
        
    
        
if __name__ == '__main__':
    dataset = pd.read_csv("data/new_data.csv")
#    w = {"energy100g":1,"saturatedfat100g":1,"sugars100g":1,"fiber100g":2,"proteins100g":2,"sodium100g":1}
    titles = ['energy100g','saturatedfat100g','sugars100g','fiber100g','proteins100g','sodium100g']
#    pi = pd.read_excel("limiting_profiles.xlsx")


#    limiting_profiles(dataset,titles)
    limiting_profiles_dist(dataset, titles)
        