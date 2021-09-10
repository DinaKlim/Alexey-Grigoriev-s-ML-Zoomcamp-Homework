# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 10:44:05 2021

@author: Dina
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv('data.txt') # read the dataset
df = pd.DataFrame(dataset) #convert to dataframe

df_BMW = df[df['Make'] == "BMW"] # select entries with "BMW" in the "Make" column

Average_BMW_price = df_BMW['MSRP'].mean()

# Question 4
df_after_2015 = df[df['Year'] >= 2015] # select all entries where Year is greater or equal 2015

missing_Engine_HP = df_after_2015['Engine HP'].isnull().sum() # how many missing values in "Engine HP"?

# calculate mean

mean_hp_before = df_after_2015['Engine HP'].mean()
df_after_2015.fillna(mean_hp_before, inplace = True)
mean_hp_after = df_after_2015['Engine HP'].mean()
print(round(mean_hp_before))
print(round(mean_hp_after))

# question 6
df_RR = df[df['Make'] == "Rolls-Royce"]
subset_RR = df_RR[['Engine HP', 'Engine Cylinders', 'highway MPG']]
dropped = subset_RR.drop_duplicates()
# get the underlying array, call it x

x = np.array(dropped)
xt = x.transpose()

# result of multiplication of x by xt
xtx = np.matmul(xt,x)
# invert xtx
xtx_inverse = np.linalg.inv(xtx)
S_xtx_inverse = sum(xtx_inverse)
Sum_S_xtx_inverse = sum(S_xtx_inverse)
print(Sum_S_xtx_inverse)

# 
y = [1000, 1100, 900, 1200, 1000, 850, 1300]
w = np.matmul(np.matmul(xtx_inverse,xt),y)
print(w[0])