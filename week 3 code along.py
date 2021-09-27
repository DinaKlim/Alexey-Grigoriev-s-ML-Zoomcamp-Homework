# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 22:45:58 2021

@author: Dina
"""

# week 3
# churn (leave) prediction project
# score each customer and see what is the likelyhood of churning
# feature vector describing customer = x
# y = vector {0, 1} 1 is positive (he did churn), 0 negative
# output of the model is the score between zero and one, the 
# likelihood of the client to churn

# we build the model using historical data and score every one of them
# target those who are likely to churn with promotion
# kaggle dataset telco
# download the data, EDA, this time with scikit learn
# we will do analysis to understand which features are important

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.txt')

df.head()

# to have a look at the dataset, transpose

df.head().T # to console

# data not uniform: lower case

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')
    
# df.dtypes # console
# total charges should be number, but it is an object. Some of the values are 
# probably not numbers
# df.totalcharges # console
# pd.to_numeric(df.totalcharges)
# not possible to convert the string "_" to number: if there is no data,
# there was a space, and spaces were replaced with underscores
# was treated as a string (object)

# if can't parce the string to number, then write NaN: errors = 'coerce'
tc = pd.to_numeric(df.totalcharges, errors='coerce')
# tc.isnull().sum()
#  df[tc.isnull()][['customerid', 'totalcharges']] console
df.totalcharges = tc
df.totalcharges = tc.fillna(0)


# let's look at churn variable: it's yes or no, but we need 0 or 1
# replace the variable with a number
df.churn = (df.churn == 'yes').astype(int)

# 3.3 Setting up validation network
# fix random seed to ensure results are reproducible
# splits into train (full train) and test
# to have three datasets, need to split full train into train and validation
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)
# len(df_full_train)
# 20% of 80% is 25%
df_train, df_val = train_test_split(df_full_train, test_size = 0.25)
# reset indices 
df_train = df_train.reset_index(drop = True)
df_val = df_train.reset_index(drop = True)
df_test = df_train.reset_index(drop = True)
# get y variable, which is churn

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

# delete the churn variable
del df_train['churn']
del df_val['churn']
del df_test['churn']

# 3.4 EDA
df_full_train = df_full_train.reset_index(drop = True)
df_full_train.isnull().sum() # to console

df_full_train.churn.value_counts(normalize=True) # to console. 
# Or use mean: if we have a binary variable, the possibilities are either 
# zeros or ones, sum of x_i is the sum of ones, divide by number of 
# observasions - get the churn rate
# number of churn users is 27%
global_churn_rate = round(df_full_train.churn.mean(),2)

# look at categorical variables
df_full_train.dtypes # to console

# create a list containing three numerical variables
numerical = ['tenure', 'monthlycharges', 'totalcharges']

df_full_train.columns # console

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
       'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']

# function n.unique will calculate number unique values within each variable
df_full_train[categorical].nunique() # console

# 3.5 Feature importance
# look at churn rates within different groups

# calculate global churn rate

global_churn = df_full_train.churn.mean()
churn_female = df_full_train[df_full_train.gender == 'female'].churn.mean()
churn_male = df_full_train[df_full_train.gender == 'male'].churn.mean()

# almost the same. Look at partner
churn_partner = df_full_train[df_full_train.partner == 'yes'].churn.mean()
churn_no_partner = df_full_train[df_full_train.partner == 'no'].churn.mean()

diff_churn_partner = churn_partner - global_churn
# instead of looking at the differences, we can divide one by another
diff_churn_partner_div = churn_partner / global_churn
# risk ratio: risk = group/global. >1 - more likely to churn, <1 - less likely to churn
# risk ratio tells the difference in relative terms and gives intuition
# about what feature can be important

# implement a cycle: gender, 
# avg(churn), diff(churn)
# from data group by gender

# get a dataframe: list of different aggregations we can perform 
df_group = df_full_train.groupby('gender').churn.agg(['mean', 'count'])
df_group['diff'] = df_group['mean'] - global_churn
df_group['risk'] = df_group['mean'] / global_churn

# for each categorical variable

for c in categorical:
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk'] = df_group['mean'] / global_churn
    display(df_group) #as we are in the loop, only with display will the results be visible
    print()
    print()

# 3.6 Mutual information
# we need a measure to say what is more important than other variable
# measure of mutual dependance: what we get about one variable 
# by observing another variable
# if we know that this particular customer has a contract, how much 
# can we tell about churn? 
# the higher the mutual information is, the more we can learn

from sklearn.metrics import mutual_info_score
mis = mutual_info_score(df_full_train.churn, df_full_train.contract)
gen = mutual_info_score(df_full_train.gender, df_full_train.churn)

# what we learned from the contract is more important than what we learn from the gender
# we can apply this metric to all the variable

def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)

mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending = False)

# 3.7 Feature importance
# Pearson's correlation coefficient: way to measure dependency between
# two variables
# R is between -1 and 1. When correlation is negative, increase in x1 leads to decrease in x2
# if the correlation is positive, increase of one value leads to increase of the other
# R between 0-01: correlation non-existent or low if r around 0.2
# R between 0.2-0.5 moderate (sometimes)
# R between 0.6 and 1: strong, i.e. often increase of one value leads to increase in the other
# Y is binary between 0 and 1, x is from -infinity to +infinity

# df_full_train.tenure.max() - check the max value

# check correlation
corr_check = df_full_train[numerical].corrwith(df_full_train.churn)
corr_check_abs = df_full_train[numerical].corrwith(df_full_train.churn).abs()
# look at different conditions and means of churn rate: people who
# stayed with the company for more than two months and less than 12

tenure_2_12 = df_full_train[(df_full_train.tenure > 2) & (df_full_train.tenure <= 12)].churn.mean()

monthly_charges_churn_below_20 = df_full_train[(df_full_train.monthlycharges < 20)].churn.mean()
monthly_charges_churn_between_20_50 = df_full_train[(df_full_train.monthlycharges > 20) & (df_full_train.monthlycharges < 50)].churn.mean()
monthly_charges_churn_over_50 = df_full_train[(df_full_train.monthlycharges > 50)].churn.mean()

# 3.8 Encode categorical features with scikit-learn

# two variables, gender and contract, all combinations (male, 1 year, female, monthly etc)
# give matrix with 5 variables: female, 1 or 0; male, 1 or 0; 
# monthly contract, 1 or 0; 1Y contract, 1 or 0; 2Y contract, 1 or 0
# hot means that the value that is activated, the value "ones" are "hot", activated;
# values that are not activated, "cold", are blue (currency flowing analogy)

from sklearn.feature_extraction import DictVectorizer

dicts = df_train[['gender','contract', 'tenure']].iloc[:100].to_dict(orient='records') #take a look at the first 10
dv = DictVectorizer(sparse = False)
# first train the vectorizer - it infers there are the columns and values, it infers what the column names are,
# what the variables there are. After adding tenure, it understood that 
# tenure is numerical and added the numerical value

dv.fit(dicts)

# dv.transform(dicts) 
# produces a sparse matrix with compressed row format

# dv.get_feature_names() # check in the console - why total charges are treated as categorical? 

# put all the variables and train
train_dict = df_train[categorical + numerical].to_dict(orient='records')
# train_dict[0] # console
# dv = DictVectorizer(sparse = False)
X_train = dv.fit_transform(train_dict)
val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

# 3.9 Logistic regression

# g(xi) is probability of xi belonging to the positive class
# similar to linear regression: g(xi) = w0 + w_t*xi
# logistic regression: instead of inputting w0 and wt between -infinity and +infinity,
# apply sigmoid and the w0 and w1 are between 0 and 1. 1/(1+exp(z))
def sigmoid(z):
    return 1/ (1 + np.exp(-z))
z = np.linspace(-5, 5, 51)    

plt.plot(z, sigmoid(z))
plt.show

def logistic_regression(xi):
    result = w0
    
    for j in range(len(w)):
        score = score + xi[j] + w[j]
    result = sigmoid(score)
    return(result)

# both linear and logistic regression are linear models: good quality,
# quite fast
       
# 3.10 Logistic regression with scikit learn
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train) 

model.coef_[0] # to console
model.intercept_[0] # to console

y_pred = model.predict_proba(X_val)[:,1] # soft predictions: probabilities, first column -
# probability of negative result, second - probability of positive result
# we are only interested in the second column
churn_decision = (y_pred >= 0.5)

df_val[churn_decision].customerid #select all customers that are likely to churn; they will
# receive a promotional email

# let's see how accurate predictions are: we used RMSE
# accuracy

churn_decision = churn_decision.astype(int)
mean_accuracy = (y_val == churn_decision).mean() # how many predictions match

df_pred = pd.DataFrame() # create dataframe
df_pred['probability'] = y_pred
df_pred['prediction'] = churn_decision.astype(int)
df_pred['actual'] = y_val

df_pred['correct'] = df_pred.prediction == df_pred.actual

# true = 1, false = 0
correct_mean = df_pred.correct.mean()

# 3.11 Model interpretation

# we have model parameters: dictionary of the features and weights using zip: 
# joining the lists together
values_variables_model = dict(zip(dv.get_feature_names(), model.coef_[0].round(3)))

# train a smaller model: take a subset of features
small = ['contract', 'tenure', 'monthlycharges']
# dicts_train_small = df_train[small].iloc[:100].to_dict(orient='records') commented because the shape of X_train_small was 4225,10
# dicts_val_small = df_val[small].iloc[:100].to_dict(orient='records')

dicts_train_small = df_train[small].to_dict(orient='records')
dicts_val_small = df_val[small].to_dict(orient='records')

dv_small = DictVectorizer(sparse = False)
dv_small.fit(dicts_train_small)
dv_small.get_feature_names()

X_train_small = dv_small.transform(dicts_train_small)
model_small = LogisticRegression()
model_small.fit(X_train_small, y_train) # fitting the model with 5 features to the 
# training y
w0 = model_small.intercept_[0] # 
w = model_small.coef_[0] # 

# 3.12 Using the model
# train the final model: we trained with all the features, 80% accuracy
# will get the big model with all the features and train it on the full train dataset

# get dictionaries
dicts_full_train = df_full_train[categorical + numerical].to_dict(orient ='records')
dv = DictVectorizer(sparse = False)
X_full_train = dv.fit_transform(dicts_full_train) 
y_full_train = df_full_train.churn.values

model = LogisticRegression()
model.fit(X_full_train, y_full_train)

# repeat the process for the test dataset
dicts_test = df_test[categorical + numerical].to_dict(orient ='records')
X_test = dv.transform(dicts_test)
y_pred = model.predict_proba(X_test)[:,1]
churn_decision = (y_pred >= 0.5)
Churn_dec_mean = (churn_decision == y_test).mean()
# accuracy difference is 1 percent, not a big deal - we have used more data
# how we can use it
# we want to know if the customer wants to leave or not
customer = dicts_test[10]
# put this array and compute the churn score
# the model computes the score and returns the probability
X_small = dv.transform([customer]) # get feature matrix
model.predict_proba(X_small)[0, 1]
y_test[10] # check in the console if he is going to churn
customer = dicts_test[-1]
X_small = dv.transform([customer])
model.predict_proba(X_small)[0, 1]
