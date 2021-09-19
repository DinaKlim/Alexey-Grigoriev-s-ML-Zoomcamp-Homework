# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 21:16:05 2021

@author: Dina
"""

import pandas as pd
import numpy as np


# New York City Airbnb Open Data: AB_NYC_2019.csv

# EDA
# load the data

df = pd.read_csv('AB_NYC_2019.csv')
df.head
# df.dtypes to see the headings

# look at distribution of price
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline # for the notebook
sns.histplot(df.price, bins = 50) # bins is how many bars we have
plt.show()

# yes it has a long tail and is skewed to the side

# choose the columns
base = ['latitude','longitude', 'price', 
        'minimum_nights', 'number_of_reviews',
        'reviews_per_month', 'calculated_host_listings_count',
        'availability_365']

df_all = df[base].values
df_all = pd.DataFrame(df_all)

# Q1. Find a feature with missing values, how many?

missing_values = df.isnull().sum() # in the console:
# result: reviews_per_month has 10052 missing values

# Q2 Median for "minimum nights"
median_price = pd.DataFrame.median(df.price)
# result: 106

# Shuffle the initial dataset, use seed 42.
# Split your data in train/val/test sets, with 60%/20%/20% distribution.
n = len(df)
n_val = int(len(df) * 0.2) 
n_test = int(len(df) * 0.2)
n_train = len(df) - n_val - n_test

# use iloc for getting part of the dataset
df_val = df.iloc[:n_val]
df_test = df.iloc[n_val:n_val+n_test]
df_train = df.iloc[n_val+n_test:]

# shuffle
idx = np.arange(n)
# shuffle indices
np.random.seed(42) # make the random reproducible
np.random.shuffle(idx)

df_train = df.iloc[idx[n_val+n_test:]]
df_val = df.iloc[idx[:n_val]]
df_test = df.iloc[idx[n_val:n_val+n_test]]

# drop the index
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

#transformation with y: apply log transform
y_train = np.log1p(df_train.price.values)
y_val = np.log1p(df_val.price.values)
y_test = np.log1p(df_test.price.values)

# Make sure that the target value ('price') is not in your dataframe.
del df_train['price']
del df_val['price']
del df_test['price']

# Q3  We need to deal with missing values for the column from Q1.
# We have two options: fill it with 0 or with the mean of this variable.
# Try both options. For each, train a linear regression model without regularization using the code from the lessons.
# For computing the mean, use the training only!
# Use the validation dataset to evaluate the models and compare the RMSE of each option.
# Round the RMSE scores to 2 decimal digits using round(score, 2)
# Which option gives better RMSE?
base = ['latitude','longitude',  
        'minimum_nights', 'number_of_reviews',
        'reviews_per_month', 'calculated_host_listings_count',
        'availability_365']
X_train = df_train[base].values

# fill with zeroes

X_train = df_train[base].fillna(0).values

def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:] # tuple with bias term and weights

w0, w = train_linear_regression(X_train, y_train)
y_pred = w0 + X_train.dot(w) # multiply feature matrix by w vector and get predictions

def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

score_filled_with_zeros = round(rmse(y_pred, y_train),2)

# # fill with means
# X_train = df_train[base].fillna(df_train[base].mean())
# w0, w = train_linear_regression(X_train, y_train)
# y_pred = w0 + X_train.dot(w) # multiply feature matrix by w vector and get predictions
# score_filled_with_means = round(rmse(y_pred, y_train),2)

# given the precision the RMSEs are the same

# Q4 Now let's train a regularized linear regression.

# For this question, fill the NAs with 0.
# Try different values of r from this list: [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10].
# Use RMSE to evaluate the model on the validation dataset.
# Round the RMSE scores to 2 decimal digits.
# Which r gives the best RMSE?
X_train = df_train[base].values
def prepare_X(df):
    df_num = df[base]  
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

X_train = prepare_X(df_train)

def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:] # tuple with bias term and 

for r in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    X_train = prepare_X(df_train)
    w0, w = train_linear_regression_reg(X_train, y_train, r = r)
    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    score = round(rmse(y_val, y_pred),2)
    print (r, w0, score)
    
# Q5 We used seed 42 for splitting the data. Let's find out how selecting the seed influences our score.
# Try different seed values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
# For each seed, do the train/validation/test split with 60%/20%/20% distribution.
# Fill the missing values with 0 and train a model without regularization.

rmse_shuffle = []

for s in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    idx = np.arange(n)
    # shuffle indices
    np.random.seed(s) # make the random reproducible
    np.random.shuffle(idx)
    
    df_train = df.iloc[idx[n_val+n_test:]]
    df_val = df.iloc[idx[:n_val]]
    df_test = df.iloc[idx[n_val:n_val+n_test]]
    
    # drop the index
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    #transformation with y: apply log transform
    y_train = np.log1p(df_train.price.values)
    y_val = np.log1p(df_val.price.values)
    y_test = np.log1p(df_test.price.values)
    
    # Make sure that the target value ('price') is not in your dataframe.
    del df_train['price']
    del df_val['price']
    del df_test['price']
    # Prepare X (fill with zeros)
    X_train = prepare_X(df_train)
    w0, w = train_linear_regression(X_train, y_train)
    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    # For each seed, evaluate the model on the validation dataset and collect the RMSE scores.
    rmse_shuffle.append(rmse(y_val, y_pred))
    print ('shuffle', s, 'score', rmse_shuffle[-1]) # -1 means the last element
    
print("Vector with rmse", rmse_shuffle)

# What's the standard deviation of all the scores? To compute the standard deviation, use np.std.
# Round the result to 3 decimal digits (round(std, 3))

std_dev_rmse_shuffle = round(np.std(rmse_shuffle),3)

# Q6 Split the dataset like previously, use seed 9.
idx = np.arange(n)
    # shuffle indices
np.random.seed(9) # make the random reproducible
np.random.shuffle(idx)

df_train = df.iloc[idx[n_val+n_test:]]
df_val = df.iloc[idx[:n_val]]
df_test = df.iloc[idx[n_val:n_val+n_test]]

# drop the index
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

#transformation with y: apply log transform
y_train = np.log1p(df_train.price.values)
y_val = np.log1p(df_val.price.values)
y_test = np.log1p(df_test.price.values)

# Make sure that the target value ('price') is not in your dataframe.
del df_train['price']
del df_val['price']
del df_test['price']

# Combine train and validation datasets.
# combine test and validation set into one
df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop = True)
X_full_train = prepare_X(df_full_train)

# combine train and val y into one y
y_full_train = np.concatenate([y_train, y_val]) # not pd but numpy; no indices, no need to drop them

# Fill the missing values with 0 and train a model with r=0.001.
def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:] # tuple with bias term and weights

X_train = prepare_X(df_full_train)
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r = 0.001)
X_test = prepare_X(df_test)
y_pred = w0 + X_test.dot(w)
rmse_reg_q6 = rmse(y_test, y_pred)
# What's the RMSE on the test dataset? 0.64