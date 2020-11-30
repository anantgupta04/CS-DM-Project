#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:41:01 2020

@author: Cata
"""

import pandas as pd
import numpy as np
from additive import PATH

dataset = pd.read_csv(PATH + "new_data.csv")
titles = ['energy100g','saturatedfat100g','sugars100g','fiber100g','proteins100g','sodium100g']

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
    titles = ['energy100g','saturatedfat100g','sugars100g','fiber100g','proteins100g','sodium100g']

    limiting_profiles_dist(dataset, titles)
        