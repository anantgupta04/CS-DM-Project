#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:47:47 2020

@author: Javi
"""

import pandas as pd
import numpy as np
from additive import PATH
import re

chars_to_remove = ['.', '-', '(', ')', '', ',']
regular_expression = '[' + re.escape (''. join (chars_to_remove)) + ']'
#df['string_col'].str.replace(regular_expression, '', regex=True)

data = pd.read_csv(PATH + "product_Germany.csv")

data['saturatedfat100g'] = data['Saturatedfat']
data['energy100g'] = data['Energy'].str.split(' ').str[0]
data['sugars100g'] = data['Sugars']
data['proteins100g'] = data['Proteins']
data['fiber100g'] = data['Fibers']
data['sodium100g'] = data['Salt']
data['nutriscorescore'] = data['Nutritionscorefrance']
data['nutriscoregrade'] = data['Nutriscore'].str.lower()
data['productname'] = data['product']

new_data = data[['productname','nutriscorescore','nutriscoregrade','energy100g','saturatedfat100g',
                 'sugars100g','fiber100g','proteins100g','sodium100g']].copy()

new_data.dropna(inplace=True)
#new_data = new_data.replace(to_replace='None', value=np.nan).dropna()
new_data = new_data.replace(to_replace='' , value=np.nan).dropna()
#new_data = new_data.replace(to_replace='<', value=0)
new_data = new_data.replace(to_replace='>', value=np.nan).dropna()
new_data = new_data.replace(to_replace='?', value=np.nan).dropna()
new_data = new_data.replace(to_replace='~', value='')
new_data.dropna(inplace=True)
new_data['energy100g'] = new_data['energy100g'].str.replace(',', '').astype(float)
new_data['fiber100g'] = new_data['fiber100g'].str.replace('~ ', '').astype(float)
new_data['saturatedfat100g'] = new_data['saturatedfat100g'].str.replace('> ', '')
new_data['saturatedfat100g'] = new_data['saturatedfat100g'].str.replace('~ ', '').astype(float)

new_data['saturatedfat100g'] = new_data['saturatedfat100g'].astype(float)
new_data['sugars100g'] = new_data['sugars100g'].astype(float)
new_data['proteins100g'] = new_data['proteins100g'].astype(float)
new_data['fiber100g'] = new_data['fiber100g'].astype(float)
new_data['sodium100g'] = new_data['sodium100g'].astype(float)

for index, values in new_data.iterrows():
    if sum(values[3:]) == 0.0:
        new_data.drop(index, inplace=True)

new_data.drop(8512, inplace=True)

new_data.to_csv('new_data.csv')