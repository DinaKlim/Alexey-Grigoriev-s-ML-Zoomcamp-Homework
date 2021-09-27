# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 16:46:17 2021

@author: Dina
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# read the NYC air BnB data AB_NYC_2019.csv
df = pd.read_csv('AB_NYC_2019.csv')

df = pd.DataFrame(df)

features = ['neighbourhood_group',
'room_type',
'latitude',
'longitude',
'price',
'minimum_nights',
'number_of_reviews',
'reviews_per_month',
'calculated_host_listings_count',
'availability_365']

df = df[features].fillna(0)

# Q1 What is the most frequent observation (mode) for the column 'neighbourhood_group'?

group = df.groupby('neighbourhood_group')
group.count() # console

# Answer: Manhattan

# Split your data in train/val/test sets, with 60%/20%/20% distribution.

# Use Scikit-Learn for that (the train_test_split function) and set the seed to 42.


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
# len(df_full_train)
# 20% of 80% is 25%
df_train, df_val = train_test_split(df_full_train, test_size = 0.25)
# reset indices 
df_train = df_train.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)

X_train = df_train[features]
X_val = df_val[features]
X_test = df_test[features]

y_train = df_train.price.values
y_val = df_val.price.values
y_test = df_test.price.values

# Make sure that the target value ('price') is not in your dataframe.

del X_train['price']
del X_val['price']
del X_test['price']

# Q 2
# Create the correlation matrix for the numerical features of your train dataset.

numerical = ['latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
             'calculated_host_listings_count', 'availability_365']
# In a correlation matrix, you compute the correlation coefficient between every pair of features in the dataset.
correlation_matrix = df[numerical].corr()
correlation_matrix.to_csv()

with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(correlation_matrix)
# What are the two features that have the biggest correlation in this dataset?

# Answer: reviews per month and number of reviews

# Make price binary

# We need to turn the price variable from numeric into binary.
# Let's create a variable above_average which is 1 if the price is above (or equal to) 152.
price_above_average_train = (y_train >= 152).astype(int)
price_above_average_val = (y_val >= 152).astype(int)
price_above_average_test = (y_test >= 152).astype(int)

# Q3 

# Calculate the mutual information score with the (binarized) price for the two categorical variables that we have. Use the training set only.
# Which of these two variables has bigger score?
# Round it to 2 decimal digits using round(score, 2)

categorical = ['neighbourhood_group', 'room_type']

from sklearn.metrics import mutual_info_score

def mutual_info_price_score(series):
    return mutual_info_score(series, df_train.price)

mi = round(df_train[categorical].apply(mutual_info_price_score),2)
mi.sort_values(ascending = False)

# answer: room_type

# Question 4
# Now let's train a logistic regression
# Remember that we have two categorical variables in the data. Include them using one-hot encoding.
dicts_train = df_train[categorical].to_dict(orient='records')
dicts_val = df_val[categorical].to_dict(orient='records')
dicts_test = df_test[categorical].to_dict(orient='records')
# Fit the model on the training dataset.
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse = False)
X_train = dv.fit_transform(dicts_train) 
y_train = price_above_average_train
X_val = dv.fit_transform(dicts_val)
# To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
    
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
model.fit(X_train, y_train) 

model.coef_[0] # to console
model.intercept_[0] # to console    

# Calculate the accuracy on the validation dataset and rount it to 2 decimal digits.
y_val = price_above_average_val

y_pred = model.predict_proba(X_val)[:,1] # soft predictions: probabilities, first column -
# probability of negative result, second - probability of positive result
# we are only interested in the second column
price_prediction = (y_pred >= 0.5).astype(int)
# y_val = pd.DataFrame(y_val)
# y_pred = pd.DataFrame(y_pred)
mean_accuracy = round((y_val == price_prediction).mean(), 2) # how many predictions match

# Question 5
# We have 9 features: 7 numerical features and 2 categorical.
# Let's find the least useful one using the feature elimination technique.
# Train a model with all these features (using the same parameters as in Q4).
dicts_train = df_train[categorical + numerical].to_dict(orient='records')
dicts_val = df_val[categorical + numerical].to_dict(orient='records')
dicts_test = df_test[categorical + numerical].to_dict(orient='records')
# Fit the model on the training dataset.
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse = False)
X_train = dv.fit_transform(dicts_train) 
y_train = price_above_average_train
X_val = dv.fit_transform(dicts_val)
# To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
    
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
model.fit(X_train, y_train) 

model.coef_[0] # to console
model.intercept_[0] # to console    

# Calculate the accuracy on the validation dataset and rount it to 2 decimal digits.
y_val = price_above_average_val

y_pred = model.predict_proba(X_val)[:,1] # soft predictions: probabilities, first column -
# probability of negative result, second - probability of positive result
# we are only interested in the second column
price_prediction = (y_pred >= 0.5).astype(int)
# y_val = pd.DataFrame(y_val)
# y_pred = pd.DataFrame(y_pred)

mean_accuracy_all_parameters = (y_val == price_prediction).mean() # how many predictions match
# Now exclude each feature from this set and train a model without it. Record the accuracy for each model.
# For each feature, calculate the difference between the original accuracy and the accuracy without the feature.
# Which of following feature has the smallest difference?
# neighbourhood_group
# room_type
# number_of_reviews
# reviews_per_month
categorical1 = ['room_type']
dicts_train = df_train[categorical1 + numerical].to_dict(orient='records')
dicts_val = df_val[categorical1 + numerical].to_dict(orient='records')
dicts_test = df_test[categorical1 + numerical].to_dict(orient='records')
# Fit the model on the training dataset.
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse = False)
X_train = dv.fit_transform(dicts_train) 
y_train = price_above_average_train
X_val = dv.fit_transform(dicts_val)
# To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
    
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
model.fit(X_train, y_train) 

# Calculate the accuracy on the validation dataset and rount it to 2 decimal digits.
y_val = price_above_average_val

y_pred = model.predict_proba(X_val)[:,1] # soft predictions: probabilities, first column -
# probability of negative result, second - probability of positive result
# we are only interested in the second column
price_prediction = (y_pred >= 0.5).astype(int)
# y_val = pd.DataFrame(y_val)
# y_pred = pd.DataFrame(y_pred)
mean_accuracy_categorical1 = (y_val == price_prediction).mean() # how many predictions match

diff_cat1 = mean_accuracy_all_parameters - mean_accuracy_categorical1

categorical2 = ['neighbourhood_group']
dicts_train = df_train[categorical2 + numerical].to_dict(orient='records')
dicts_val = df_val[categorical2 + numerical].to_dict(orient='records')
dicts_test = df_test[categorical2 + numerical].to_dict(orient='records')
# Fit the model on the training dataset.
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse = False)
X_train = dv.fit_transform(dicts_train) 
y_train = price_above_average_train
X_val = dv.fit_transform(dicts_val)
# To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
    
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
model.fit(X_train, y_train) 

# Calculate the accuracy on the validation dataset and rount it to 2 decimal digits.
y_val = price_above_average_val

y_pred = model.predict_proba(X_val)[:,1] # soft predictions: probabilities, first column -
# probability of negative result, second - probability of positive result
# we are only interested in the second column
price_prediction = (y_pred >= 0.5).astype(int)
# y_val = pd.DataFrame(y_val)
# y_pred = pd.DataFrame(y_pred)
mean_accuracy_categorical2 = (y_val == price_prediction).mean() # how many predictions match
diff_cat2 = mean_accuracy_all_parameters - mean_accuracy_categorical2

numerical1 = ['latitude', 'longitude', 'minimum_nights', 'reviews_per_month',
             'calculated_host_listings_count', 'availability_365']
dicts_train = df_train[categorical + numerical1].to_dict(orient='records')
dicts_val = df_val[categorical + numerical1].to_dict(orient='records')
dicts_test = df_test[categorical + numerical1].to_dict(orient='records')
# Fit the model on the training dataset

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse = False)
X_train = dv.fit_transform(dicts_train) 
y_train = price_above_average_train
X_val = dv.fit_transform(dicts_val)
# To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
    
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
model.fit(X_train, y_train) 

# Calculate the accuracy on the validation dataset and rount it to 2 decimal digits.
y_val = price_above_average_val

y_pred = model.predict_proba(X_val)[:,1] # soft predictions: probabilities, first column -
# probability of negative result, second - probability of positive result
# we are only interested in the second column
price_prediction = (y_pred >= 0.5).astype(int)
# y_val = pd.DataFrame(y_val)
# y_pred = pd.DataFrame(y_pred)
mean_accuracy_numerical1 = (y_val == price_prediction).mean() # how many predictions match
diff_cat3 = mean_accuracy_all_parameters - mean_accuracy_numerical1

numerical2 = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 
             'calculated_host_listings_count', 'availability_365']
dicts_train = df_train[categorical + numerical2].to_dict(orient='records')
dicts_val = df_val[categorical + numerical2].to_dict(orient='records')
dicts_test = df_test[categorical + numerical2].to_dict(orient='records')
# Fit the model on the training dataset

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse = False)
X_train = dv.fit_transform(dicts_train) 
y_train = price_above_average_train
X_val = dv.fit_transform(dicts_val)
# To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
    
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
model.fit(X_train, y_train) 

# Calculate the accuracy on the validation dataset and rount it to 2 decimal digits.
y_val = price_above_average_val

y_pred = model.predict_proba(X_val)[:,1] # soft predictions: probabilities, first column -
# probability of negative result, second - probability of positive result
# we are only interested in the second column
price_prediction = (y_pred >= 0.5).astype(int)
# y_val = pd.DataFrame(y_val)
# y_pred = pd.DataFrame(y_pred)
mean_accuracy_numerical2 = (y_val == price_prediction).mean() # how many predictions match
diff_cat4 = mean_accuracy_all_parameters - mean_accuracy_numerical2

# answer: diff_cat2 is smallest, parameter = 'room_type'

# Question 6
# For this question, we'll see how to use a linear regression model from Scikit-Learn
# We'll need to use the original column 'price'. Apply the logarithmic transformation to this column.
df.price = np.log1p(df.price)
# Fit the Ridge regression model on the training data.
features = ['neighbourhood_group',
'room_type',
'latitude',
'longitude',
'price',
'minimum_nights',
'number_of_reviews',
'reviews_per_month',
'calculated_host_listings_count',
'availability_365']

df = df[features].fillna(0)


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
# len(df_full_train)
# 20% of 80% is 25%
df_train, df_val = train_test_split(df_full_train, test_size = 0.25)
# reset indices 
df_train = df_train.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)

X_train = df_train[features]
X_val = df_val[features]
X_test = df_test[features]

y_train = df_train.price.values
y_val = df_val.price.values
y_test = df_test.price.values

# Make sure that the target value ('price') is not in your dataframe.

del X_train['price']
del X_val['price']
del X_test['price']

dicts_train = df_train[categorical].to_dict(orient='records')
dicts_val = df_val[categorical].to_dict(orient='records')
dicts_test = df_test[categorical].to_dict(orient='records')

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse = False)
X_train = dv.fit_transform(dicts_train) 
X_val = dv.fit_transform(dicts_val)
    
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
val_error1 = []

alpha = [0, 0.01, 0.1, 1, 10]
for a in alpha:
    model = Ridge(alpha = a, solver = 'svd')
    model.fit(X_train, y_train) 
    y_predict = model.predict(X_val)
    val_error1.append(mean_squared_error(y_val, y_predict))
    print(a, round(mean_squared_error(y_val, y_predict),3))
# This model has a parameter alpha. Let's try the following values: [0, 0.01, 0.1, 1, 10]
# Which of these alphas leads to the best RMSE on the validation set? Round your RMSE scores to 3 decimal digits.
# If there are multiple options, select the smallest alpha.
# answer: 0 