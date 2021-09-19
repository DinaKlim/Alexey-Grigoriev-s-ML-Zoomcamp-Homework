# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 14:36:38 2021

@author: Dina
"""

import numpy as np
import pandas as pd

# video 2.2
df = pd.read_csv('data.csv')
df.head

# replace spaces with underscores
df.columns = df.columns.str.lower().str.replace(' ', '_')
# define string columns
df.dtypes
# objects are interesting to us
# get columns of type object
strings = list(df.dtypes[df.dtypes == 'object'].index)
# loop over them, replace the space with underscore and 
# write the value back into the column
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')
    
# video 2.3 Exploratory data analysis
for col in df.columns:
    print(col)
    print(df[col].head())
    print()
    
# print first five unique values
for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())
    print()
    
# look at distribution of price
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline # for the notebook
sns.histplot(df.msrp, bins = 50) # bins is how many bars we have
plt.show()
# most cars are between 0 and 0.3 - long tail distribution
# maybe one car costs 2M, 1.5 M

# zoom in
# sns.histplot(df.msrp[df.msrp < 100000], bins = 50)

# the long tail will confuse the model 
# get rid of the tail: apply logarithm
# problems with lg: for zero it does not exist, 
# avoid the problem by adding 1: log1p (1p = plus one)

price_logs = np.log1p(df.msrp)
sns.histplot(price_logs, bins = 50)
plt.show()
# the shape resembles the double shaped bell: ideal for models

# look at the missing values (nan)

df.isnull().sum() #will probably work in jupiter? 

# 2.4 settting up validation framework
# calculate how many are 20% of the dataset
n = len(df)
n_val = int(len(df) * 0.2) 
n_test = int(len(df) * 0.2)
n_train = len(df) - n_val - n_test

# use iloc for getting part of the dataset
df_val = df.iloc[:n_val]
df_test = df.iloc[n_val:n_val+n_test]
df_train = df.iloc[n_val+n_test:]

# dataset is sequential, need to shuffle
# generate sequence
idx = np.arange(n)
# shuffle indices
np.random.seed(2) # make the random reproducible
np.random.shuffle(idx)

df_train = df.iloc[idx[n_val+n_test:]]
df_val = df.iloc[idx[:n_val]]
df_test = df.iloc[idx[n_val:n_val+n_test]]

# drop the index
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

#transformation with y: apply log transform
y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

# delete the variables - otherwise target variable can get into the x of the train database
del df_train['msrp']
del df_val['msrp']
del df_test['msrp']

# 2.5 Linear regression
# capital X - entire feature matrix; x - one observation
# take 3 x variables and put it into the feature matrix
xi = [275,13,1385]
def g(xi):
    # do something
    return 10000
# this would predict that the price is 10000 based on the x input
# w0 bias term: prediction that we make without knowing anything about the car
# g(xi) = w0 + w1x1 + w2x2 + w3x3
# g(xi) = w0 + sum(wj*xj)

w0 = 0
# weight for each feature: wj
w = [1, 1, 1]

def linear_regression(xi):
    n = len(xi)
    pred = w0
    for j in range(n):
        pred = pred + w[j]*xi[j]
        return pred
linear_regression(xi)

# let's change weights and the bias factor
w0 = 7.17
# weight for each feature: wj
w = [0.01, 0.04, 0.002]

def linear_regression(xi):
    n = len(xi)
    pred = w0
    for j in range(n):
        pred = pred + w[j]*xi[j]
        return pred
linear_regression(xi)

# we did log transform: we need to undo the log by doing the exponent
prediction_exponent = np.exp(linear_regression(xi))

# 2.6 linear regression in vector form! 
# how to go to capital notation
# g(xi) = w0 + sum(xij*wj) = w0 + xi_transpose*w

# define function dot product
def dot(xi, w):
    n = len(xi)
    
    res = 0.0
    
    for j in range(n):
        res = res + xi[j]*w[j]
        
    return res

# simplify the linear regression with the dot product
def linear_regression(xi):
    return w0 + dot(xi, w)

# how to make it even shorter
# w = [wo w1 ... wn] size of vector is n+1
# xi = [x_io x_i1 x_i2 ... x_in]
# w_transpose*x_i = x_i_transpose * w
# first weight (x0) for bias w0 is always 1
# now can use dot product for the entire linear regression

w_new = [w0] + w # new vector with weights

# pre-pend 1 in the beginning of the x vector
def linear_regression(xi):
    xi = [1] + xi # this makes the vector xi longer by adding 1 at the beginning
    return dot(xi, w_new)
linear_regression(xi)

# X matrix
# row 1 [1 x11 x12 ... x1n] * [w0]
# row 2 [1 x21 x22 ... x2n] * [w1]
# ...
# row m [1 xm1 xm2 ... xmn] * [wm]
# the result will be prediction vector y = [x1_transpose * w x2_transpose*w ...]
# need to do matrix*vector multiplication
# implement: 
    
x1 = [1, 148, 24, 1385]
x2 = [1, 132, 25, 2031]
x10 = [1, 453, 11, 86]

X = [x1, x2, x10]
# x becomes the list of lists, make it into array
X = np.array(X)

def linear_regression(X):
    return X.dot(w_new)

# 2.7 Training a linear regression model
# we want prediction to be close to y
# if the inverse exists, w = X^(-1)*y
# however, the X is rectangular. For this matrix the inverse does not exist
# XtX = gram matrix, is square, the inverse exists, dimensions (n+1)*(n+1)
# multiply both sides by inverse of XtX
# w = (XtX)^(-1)*X_transpose*y
# w is not the solution to the system because the solution does not exist,
# but it is the closest possible solution

def train_linear_regression(X, y):
    pass
X =  [
     [148, 24, 1385],
     [132, 25, 2031],
     [453, 11, 86],
     [148, 24, 1385],
     [132, 25, 2031],
     [453, 11, 86],
     [148, 24, 1385],
     [142, 25, 431],
     [453, 11, 86],
]

X = np.array(X)

# calculate gramm-matrix
XTX = X.T.dot(X)

# find the inverse
XTX_inv = np.linalg.inv(XTX)

# check if the product of the XTX.dot and its inverse gives the identity matrix
# the precision is finite, so the numbers off the diagonal can be small numbers and not zeros
XTX_i = XTX.dot(XTX_inv)

# multiply the XTX_inv by transpose and by y
# define y

y = [100, 200, 150, 250, 100, 200, 150, 250, 120]

w = XTX_inv.dot(X.T).dot(y)

# add bias term that gives us the baseline: column of ones (commented later bacuse ones went into the function)
# ones = np.ones(X.shape[0]) # looks at number of rows and creates vectors of ones

# X = np.column_stack([ones, X])

# repeat gramm-matrix and inverse calculation because now dimensions are different
XTX = X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)

# now w contains all the weights and is called w_full
w_full = XTX_inv.dot(X.T).dot(y)

w0 = w_full[0]
w = w_full[1:]

# put everything into the function now
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:] # tuple with bias term and weights
train_linear_regression(X,y)

# 2.8 Baseline model for car price prediction
# build the model based on five features

# df_train.columns - put it in the command line
base = ['engine_hp','engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']

X_train = df_train[base].values
# df_train[base].isnull().sum() # put this in the command line to see where the missing values are

# the easiest way is to full NANs with zeros
X_train = df_train[base].fillna(0).values

# g(xi) = w0 + xi1*w1 + xi2*w2; by assuming the missing value is zero, we pretend it doesn't exist
# sometimes replacing the missing values with zeros does not make sense,
# but still works fine with ML models

w0, w = train_linear_regression(X_train, y_train)
y_pred = w0 + X_train.dot(w) # multiply feature matrix by w vector and get predictions

# plot predictions to see if they look similar to the dataset
sns.histplot (y_pred, color = 'red', alpha = 0.5, bins = 50)
sns.histplot(y_train, color = 'blue', alpha = 0.5, bins = 50)
plt.show()

# shape of predictions is off: the peaks of distribution don't match. 
# predicts smaller values than those in the dataset

# need the metrics to say objectively evaluate the performance of regression models

# 2.9 Root mean squared error
# RMSE is a difference between each prediction that we make, or g(x_i) and the actual value, or y_i
# square and take the average
# RMSE = sqrt(1/m*sum(g(x_i) - y_i)^2)
def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)
rmse(y_train, y_pred)

# 2.10 validating the model
# take the dataset, split into 3 parts
# we took the training dataset, trained a linear regression model
# then applied to the training data and calculated the error
# should apply to validation set instead
base = ['engine_hp','engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']

X_train = df_train[base].values
def prepare_X(df):
    df_num = df[base]  
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred)

# 2.11 feature engineering! Improve the model using the feature engineering
# we should use year: the younger the car, the more expensive
# look when the dataset was collected
# df_train.year.max() # copy to command line

# calculate the age of the car
2017 - df_train.year

def prepare_X(df):
    df = df.copy()
    df['age'] = 2017 - df.year
    features = base + ['age']
    
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

X_train = prepare_X(df_train)

# it changed the data: would prefer if the function did not modify; hence the df = df.copy above
w0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse_without_doors = rmse(y_val, y_pred) # copy to the command line and see that rmse has improved

# plot the predictions to see how it has improved 
sns.histplot (y_pred, color = 'red', alpha = 0.5, bins = 50)
sns.histplot(y_val, color = 'blue', alpha = 0.5, bins = 50)
plt.show()

# peaks match, generally got better, still room for improvement 

# 2.12 categorical variables: strings, not numbers (make, model, types of engine etc)
# type "object"
# one value that looks like numerical - number of doors, but it is in fact categorical variable
# we want to use categorical values: cars with 2 doors could be more expensive
# than cars with 4 doors

# we represent it with a bunch of binary values:
# instead of "number of doors = 2", introduce several columns with 
# "number of doors = 1", "number of doors = 2", "number of doors = 3"

# df_train.number_of_doors == 2

# need to write into dataframe, %s is a template where %s will be replaced by v
# (number of doors through which the loop iterates)
for v in [2, 3, 4]:
    df_train['num_doors_%s' %v] = (df_train.number_of_doors == v).astype('int')

# modify X function
def prepare_X(df):
    df = df.copy()
    df['age'] = 2017 - df.year
    features = base + ['age']
    for v in [2, 3, 4]:
        df['num_doors_%s' %v] = (df_train.number_of_doors == v).astype('int')
        features.append('num_doors_%s' %v) # add new features: akl features from the base, plus age, plus three columns with door numbers    
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

X_prepared = prepare_X(df_train) # X is not getting updated unless I use a new variable

X_train = prepare_X(df_train)

# it changed the data: would prefer if the function did not modify; hence the df = df.copy above
w0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse_with_doors = rmse(y_val, y_pred) 

# df.make,value_counts().head() # copy into the command line
# make a list from values of "makes"
makes = list(df.make.value_counts().head().index)

def prepare_X(df):
    df = df.copy()
    df['age'] = 2017 - df.year
    features = base + ['age']
    for v in [2, 3, 4]:
        df['num_doors_%s' %v] = (df.number_of_doors == v).astype('int')
        features.append('num_doors_%s' %v) # add new features: akl features from the base, plus age, plus three columns with door numbers    
    for v in makes:
        df['make_%s' %v] = (df.make == v).astype('int')
        features.append('make_%s' %v)
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

# see if the model has improved because now we have makes in the dataset
X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse_with_makes = rmse(y_val, y_pred)

# include more categorical variables: df_train.dtypes in the command line, 
# copy the variables from there into the list
# make a dictionary with all categorical variables
categorical_variables = ['make', 'engine_fuel_type', 'market_category', 'vehicle_size', 'vehicle_style']

categories = {}
for c in categorical_variables:
    categories[c] = list(df[c].value_counts().head().index) #gives the most popular five 
    
# now there will be two loops: for each categorical variable
# and inside - for each of the five most popular categories in the categorical variable

def prepare_X(df):
    df = df.copy()
    df['age'] = 2017 - df.year
    features = base + ['age']
    for v in [2, 3, 4]:
        df['num_doors_%s' %v] = (df.number_of_doors == v).astype('int')
        features.append('num_doors_%s' %v) # add new features: akl features from the base, plus age, plus three columns with door numbers    
    for c, values in categories.items():
        for v in values:
            df['%s_%s' % (c,v)] = (df[c] == v).astype('int')
            features.append('%s_%s' % (c,v))
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse_with_categories = rmse(y_val, y_pred)

# now the RMSE is huge! 

# 2.13 Regularization
# w = (XtX)^(-1)*Xtranspose*y
# sometimes features in the matrix are duplicate; in this case the inverse of XtX does not exist
# one column is expressed through other columns: matrix is singular, cannot compute the inverse
# if the matrix is invertible, but not quite (columns are the same but there are tiny differences)
# - then the numbers in the inverse matrix are huge
# can solve the problem by adding the number to the diagonal
# XTX = XTX + 0.01 * np.eye(3)
# controlling the weights so they don't grow too much: regularization

def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:] # tuple with bias term and weights

X_train = prepare_X(df_train)
w0, w = train_linear_regression_reg(X_train, y_train, r = 0.01)
X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse_with_categories_reg = rmse(y_val, y_pred)

# plot the predictions to see how it has improved 
sns.histplot (y_pred, color = 'red', alpha = 0.5, bins = 50)
sns.histplot(y_val, color = 'blue', alpha = 0.5, bins = 50)
plt.show()

# 2.14 Tuning the model
# using validation set for trying different values for r and print the results

for r in [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 10]:
    X_train = prepare_X(df_train)
    w0, w = train_linear_regression_reg(X_train, y_train, r = r)
    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    print (r, w0, score)
    
# the model hasn't started to degrade in performance, 
# and the bias term is not too big, not too small: we choose r = 0.001

# 2.15 Train the final model and use it
# Full train will be trained on the train + val set, and test it on the 
# test set, make sure it works fine and will calculate RMSE

# combine test and validation set into one
df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop = True)
X_full_train = prepare_X(df_full_train)

# combine train and val y into one y
y_full_train = np.concatenate([y_train, y_val]) # not pd but numpy; no indices, no need to drop them

# train the model
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r = 0.001)

# look at weights: w in the command line
X_test = prepare_X(df_test)
y_pred = w0 + X_test.dot(w)
rmse_with_categories_reg_final = rmse(y_test, y_pred)
# rmse hasn't changed much

# now we can use the model: extract features, get feature vector from the database
# and then predict the price

# df_test.iloc[20]
# when we extract features, we make dictionary
car = df_test.iloc[20].to_dict() # data about one car into the dictionary
df_small = pd.DataFrame([car]) # a list of dictionaries 
X_small = prepare_X(df_small)
y_pred = w0 + X_small.dot(w)
y_pred = y_pred[0] # we need only the first row, there is just one

price_predicted_exp = np.expm1(y_pred)

actual_price = np.expm1(y_test[20])