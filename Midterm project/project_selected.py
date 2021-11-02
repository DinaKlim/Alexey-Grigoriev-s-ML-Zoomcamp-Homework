# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 10:51:03 2021

@author: Dina
"""

# imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyrolite.plot import pyroplot
import plotly.express as px
import ternary
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction import DictVectorizer
# some values are below detection limit; these will be replaced with half of the detection limit
# there are occasional Xs instead of a number; these will be replaced by zeros

def item_cleanup(s):
    if type(s) != str:
        c = pd.to_numeric(s)            
    elif s == "X":
        c = 0
    elif "<" in s:
        c = pd.to_numeric(s[1:]) / 2
    else:
        c = pd.to_numeric(s)
    return c

def data_cleanup(s):
    c = s.copy()
    for i in range(s.size):
        c.values[i] = item_cleanup(s.values[i])            
    return c

# read the data
df = pd.read_csv('greenstone_geochemistry_rock_selected_groups.csv',encoding='cp1252')
# 1. EDA
# We will look at the Cu, Au, Ni (key ore elements), As and SO3 (maybe pathfinder elements, or elements that are 
# contained in the ore minerals, but are not useful by themselves):  
els = ['SO3_pct',
'Au_ppb',
'Ni_ppm',
'Cu_ppm',
'As_ppm']


ore_els = df[els].copy()
ore_els = ore_els.dropna()

ore_els = ore_els.apply(data_cleanup) # convert to numeric values

sns.histplot(ore_els['SO3_pct'], bins = 50) # bins is how many bars we have
plt.show()
sns.histplot(ore_els['Au_ppb'], bins = 50) # bins is how many bars we have
plt.show()
sns.histplot(ore_els['Ni_ppm'], bins = 50) # bins is how many bars we have
plt.show()
sns.histplot(ore_els['Cu_ppm'], bins = 50) # bins is how many bars we have
plt.show()
sns.histplot(ore_els['As_ppm'], bins = 50) # bins is how many bars we have
plt.show()

# all long tails except Au (very few significant values) and Ni (sort of bimodal)
# will need to apply log-transform

# 1.2 Basic plots for looking at rock composition (from Scott Halley's tutorial). 

# plot the density plots of Si vs Mg, Fe, Cr, Al, Sc, Ti (page 4 of the tutorial)

silica_els = ['SiO2_pct',
                'TiO2_pct',
                'Al2O3_pct',
                'Fe2O3T_pct',
                'MgO_pct',
                'Sc_ppm',
                'Cr_ppm',
                'MnO_pct',
                'CaO_pct',
                'Na2O_pct',
                'K2O_pct',
                'P2O5_pct',
                'SO3_pct']

silica_et_al = df[silica_els].copy()
silica_et_al = silica_et_al.dropna()
silica_et_al = silica_et_al.apply(data_cleanup)

# multiple detection limits! leave for the moment, but for the future this has to be half of the lowest detection limit?

# Convert oxides to els by dividing by the molar weight of oxide and multiplyting by molar weight of element:
# Mg = MgO/40.3044*24.305
# Fe = Fe2O3/159.69*55.845
# Al = Al2O3/101.96*26.9815
# Ti = TiO2/79.866*47.867
silica_et_al['Mg'] = silica_et_al['MgO_pct']/40.3044*24.305
silica_et_al['Fe'] = silica_et_al['Fe2O3T_pct']/159.69*55.845
silica_et_al['Al'] = silica_et_al['Al2O3_pct']/101.96*26.9815
silica_et_al['Ti'] = silica_et_al['TiO2_pct']/79.866*47.867

# plot with pyrolite library
fig, ax = plt.subplots(2, 3, figsize=(14, 8))       # create a composite plot with 6 figures, 3 in 2 rows
ax = ax.flat                                       # this iterates through the figures... 

Mg_vs_Si = ['SiO2_pct', 'Mg']
Fe_vs_Si = ['SiO2_pct', 'Fe']
Cr_vs_Si = ['SiO2_pct', 'Cr_ppm']
Al_vs_Si = ['SiO2_pct', 'Al']
Sc_vs_Si = ['SiO2_pct', 'Sc_ppm']
Ti_vs_Si = ['SiO2_pct', 'Ti']

silica_et_al.loc[:, Mg_vs_Si].pyroplot.density(ax=ax[0])
silica_et_al.loc[:, Fe_vs_Si].pyroplot.density(ax=ax[1])
silica_et_al.loc[:, Cr_vs_Si].pyroplot.density(ax=ax[2])
silica_et_al.loc[:, Al_vs_Si].pyroplot.density(ax=ax[3])
silica_et_al.loc[:, Sc_vs_Si].pyroplot.density(ax=ax[4])
silica_et_al.loc[:, Ti_vs_Si].pyroplot.density(ax=ax[5])

titles = ["SiO2,% vs Mg, %", "SiO2,% vs Fe, %", "SiO2,% vs Cr, ppm", "SiO2,% vs Al, %", "SiO2,% vs Sc, ppm", "SiO2,% vs Ti, %"]
for t, a in zip(titles + [i + " (log-log)" for i in titles], ax):    # put headers on the diagrams - leaving this pyroplot cycle for future log-log plots
    a.set_title(t)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 3, figsize=(14, 8))       # create a composite plot with 6 figures, 3 in 2 rows
ax = ax.flat                                       # this iterates through the figures... 
contours = [0.95, 0.7, 0.5, 0.3]  # set the density contours: cumulative 95% ... 30% of points

silica_et_al.loc[:, Mg_vs_Si].pyroplot.density(ax=ax[0], contours=contours)
silica_et_al.loc[:, Fe_vs_Si].pyroplot.density(ax=ax[1], contours=contours)
silica_et_al.loc[:, Cr_vs_Si].pyroplot.density(ax=ax[2], contours=contours)
silica_et_al.loc[:, Al_vs_Si].pyroplot.density(ax=ax[3], contours=contours)
silica_et_al.loc[:, Sc_vs_Si].pyroplot.density(ax=ax[4], contours=contours)
silica_et_al.loc[:, Ti_vs_Si].pyroplot.density(ax=ax[5], contours=contours)

for t, a in zip(titles + [i + " (log-log)" for i in titles], ax):    # put headers on the diagrams - leaving this pyroplot cycle for future log-log plots
    a.set_title(t)
plt.tight_layout()
plt.show()


# 1.2 Jensen plot  
# calculate cation percentage for Fe+Ti, Al and Mg, normalize to sum of those three to plot on a ternary diagram
silica_et_al['cat_Si'] = silica_et_al['SiO2_pct']/60.09*1
silica_et_al['cat_Ti'] = silica_et_al['TiO2_pct']/79.9*1
silica_et_al['cat_Al'] = silica_et_al['Al2O3_pct']/101.96*2
silica_et_al['cat_Fe'] = silica_et_al['Fe2O3T_pct']/159.69*2
silica_et_al['cat_Mg'] = silica_et_al['MgO_pct']/40.3*1
silica_et_al['cat_Mn'] = silica_et_al['MnO_pct']/70.94*1
silica_et_al['cat_Ca'] = silica_et_al['CaO_pct']/56.08*1
silica_et_al['cat_Na'] = silica_et_al['Na2O_pct']/61.98*2
silica_et_al['cat_K'] = silica_et_al['K2O_pct']/94.2*2
silica_et_al['cat_P'] = silica_et_al['P2O5_pct']/141.95*2
silica_et_al['cat_S'] = silica_et_al['SO3_pct']/80.06*1

silica_et_al['cat_total'] = silica_et_al['cat_Si'] + silica_et_al['cat_Ti'] + silica_et_al['cat_Al'] + silica_et_al['cat_Fe'] + \
    silica_et_al['cat_Mg'] + silica_et_al['cat_Mn'] + silica_et_al['cat_Ca'] + silica_et_al['cat_Na'] + silica_et_al['cat_K'] + \
    silica_et_al['cat_P'] + silica_et_al['cat_S'] 
    
silica_et_al['cat_total_ternary'] = silica_et_al['cat_Al'] + silica_et_al['cat_Mg'] + silica_et_al['cat_Fe'] + silica_et_al['cat_Ti']

silica_et_al['cat_Fe_Ti_ternary'] = (silica_et_al['cat_Ti'] + silica_et_al['cat_Fe'])/silica_et_al['cat_total_ternary']
silica_et_al['cat_Mg_ternary'] = silica_et_al['cat_Mg']/silica_et_al['cat_total_ternary']
silica_et_al['cat_Al_ternary'] = silica_et_al['cat_Al']/silica_et_al['cat_total_ternary']

# make a ternary plot based on: https://colab.research.google.com/github/agile-geoscience/xlines/blob/master/notebooks/12_Ternary_diagrams.ipynb#scrollTo=NuDIFksmNObR

fig, tax = ternary.figure(scale=1)
fig.set_size_inches(5, 4.5)
# tax.horizontal_line(16)
# draw lines (Rollinson, page 62)
# p1 = (22, 8, 10)
# p2 = (2, 22, 16)
# tax.line(p1, p2, linewidth=3., marker='s', color='green', linestyle=":")

tax.scatter(silica_et_al[['cat_Mg_ternary', 'cat_Fe_Ti_ternary', 'cat_Al_ternary']].values, marker='o')
tax.gridlines(multiple=10)
tax.get_axes().axis('off')
# axis labels
fontsize = 12
offset = 0.14
tax.set_title("Jensen plot attempt\n", fontsize=fontsize)
tax.right_corner_label("Mg cation %", fontsize=fontsize)
tax.top_corner_label("Ti + Fe (total) cation %", fontsize=fontsize)
tax.left_corner_label("Al cation %", fontsize=fontsize)

# make a density plot, or a heatmap (color by Cu content)
# draw lines (Rollinson, page 62) - work in progress, doesn't want to plot
# p1 = (22, 8, 10)
# p2 = (2, 22, 16)
# tax.line(p1, p2, linewidth=3., marker='s', color='green', linestyle=":")
# p1, p2 = (90, 10,), (53.5, 28.5)
# tax.line(p1, p2, linewidth=3, color='k', alpha=0.35, linestyle="-")
# p1, p2 = (75, 25, 0), (75, 0, 25)
# tax.line(p1, p2, linewidth=3, color='k', alpha=0.35, linestyle="-")
# 90, 10, 0; 53.5, 28.5, 18; 52.5, 29,
# 18.5; 51.5, 29, 19.5; 50.5, 27.5, 22;
# 50.3, 25, 24.7; 50.8, 20, 29.2; 51.5,
# 12.5, 36

plt.savefig('ternary.jpg', dpi = 300)

# select the data that is likely to be of interest: all major elements, plus some indicator elements
# we test whether we can predict Cu and Ni from other selected elements

# define list of elements for each dataset separately 

Cu_and_els = ['SiO2_pct',
         'TiO2_pct',
         'Al2O3_pct',
         'Fe2O3T_pct',
         'MgO_pct',
         'Sc_ppm',
         'Cr_ppm',
         'MnO_pct',
         'CaO_pct',
         'Na2O_pct',
         'K2O_pct',
         'P2O5_pct',
         'SO3_pct',
         'Sr_ppm',
         'Zr_ppm',
         'Y_ppm',
         'V_ppm',
         'Hf_ppm',
         'Ba_ppm',
         'Nb_ppm',
         'Cu_ppm',
         'Preferred chemical classification']

Ni_and_els = ['SiO2_pct',
         'TiO2_pct',
         'Al2O3_pct',
         'Fe2O3T_pct',
         'MgO_pct',
         'Sc_ppm',
         'Cr_ppm',
         'MnO_pct',
         'CaO_pct',
         'Na2O_pct',
         'K2O_pct',
         'P2O5_pct',
         'SO3_pct',
         'Sr_ppm',
         'Zr_ppm',
         'Y_ppm',
         'V_ppm',
         'Hf_ppm',
         'Ba_ppm',
         'Nb_ppm',
         'Ni_ppm']


# clean up data and convert to numerical
df_Cu = df[Cu_and_els].copy().dropna()
for i in Cu_and_els[:-1]:
    df_Cu[i] = df_Cu[i].apply(item_cleanup)

# df_Ni = df[Ni_and_els].copy()
# df_Ni = df_Ni.dropna()
# df_Ni = df_Ni.apply(data_cleanup)

# Split the dataset into training, validation and test sets. We should not really do it like that in 
# geology, because the datasets are often not balanced and we need another kind of split that takes into
# account rock types

from sklearn.model_selection import train_test_split
df_full_train, df_test = train_test_split(df_Cu, test_size = 0.2, random_state = 1)

# convert objects to numerical
# we still have "objects" instead of numeric values in the dataset, I have no clue why
# so I convert them here explicitly

df_full_train['Cu_ppm'] = df_full_train['Cu_ppm'].astype(float)
df_full_train['Sc_ppm'] = df_full_train['Sc_ppm'].astype(float)
df_full_train['Cr_ppm'] = df_full_train['Cr_ppm'].astype(float)
df_full_train['MnO_pct'] = df_full_train['MnO_pct'].astype(float)
df_full_train['CaO_pct'] = df_full_train['CaO_pct'].astype(float)
df_full_train['Na2O_pct'] = df_full_train['Na2O_pct'].astype(float)
df_full_train['K2O_pct'] = df_full_train['K2O_pct'].astype(float)
df_full_train['SO3_pct'] = df_full_train['SO3_pct'].astype(float)
df_full_train['Sr_ppm'] = df_full_train['Sr_ppm'].astype(float)
df_full_train['Zr_ppm'] = df_full_train['Zr_ppm'].astype(float)
df_full_train['V_ppm'] = df_full_train['V_ppm'].astype(float)
df_full_train['Hf_ppm'] = df_full_train['Hf_ppm'].astype(float)
df_full_train['Ba_ppm'] = df_full_train['Ba_ppm'].astype(float)
df_full_train['Nb_ppm'] = df_full_train['Nb_ppm'].astype(float)

df_train, df_val = train_test_split(df_full_train, test_size = 0.25)

# reset indices 
df_train = df_train.reset_index(drop = True)
df_val = df_train.reset_index(drop = True)
df_test = df_train.reset_index(drop = True)

y_train = df_train.Cu_ppm.values
y_val = df_val.Cu_ppm.values
y_test = df_test.Cu_ppm.values

# just in case our Cu long tail is real, we do a log transform and see if it changes the RMSE

y_train_log = np.log1p(df_train.Cu_ppm).values
y_val_log = np.log1p(df_val.Cu_ppm).values
y_test_log = np.log1p(df_test.Cu_ppm).values

# delete the target variable - could cause overfitting if occasionally used for training

del df_train['Cu_ppm']
del df_val['Cu_ppm']
del df_test['Cu_ppm']
# 1.2 Feature importance

# check correlation
X_columns = ['SiO2_pct',
         'TiO2_pct',
         'Al2O3_pct',
         'Fe2O3T_pct',
         'MgO_pct',
         'Sc_ppm',
         'Cr_ppm',
         'MnO_pct',
         'CaO_pct',
         'Na2O_pct',
         'K2O_pct',
         'P2O5_pct',
         'SO3_pct',
         'Sr_ppm',
         'Zr_ppm',
         'Y_ppm',
         'V_ppm',
         'Hf_ppm',
         'Ba_ppm',
         'Nb_ppm']

# check the correlation between each input element and the output element, Cu
corr_check = df_full_train[X_columns].corrwith(df_full_train.Cu_ppm)

# None of the elements are highly correlated with Cu

# Logistic regression with scikit learn
from sklearn.linear_model import LinearRegression
model_LR = LinearRegression()
X_train = df_train[X_columns]
X_val = df_val[X_columns]
X_test = df_test[X_columns]
model_LR.fit(X_train, y_train)
y_pred = model_LR.predict(X_val)
rmse_LR = np.sqrt(mean_squared_error(y_val, y_pred))

# rmse of 65 for predicting ppms of Cu means that whenever we are predicting Cu values from other elements with 
# linear regression, we will be on average 65 ppm off from the true value for the validation dataset
# plot the predictions  
plt.figure()
sns.histplot (y_pred, color = 'red', alpha = 0.5, bins = 50)
sns.histplot(y_val, color = 'blue', alpha = 0.5, bins = 50)
plt.xlim(0, 500)
plt.show()
# from the figure we see that the predictions are kind of off, and Cu had a long tail, so we will test on the log-transformed variable
 
model_LR_log = LinearRegression()
X_train = df_train[X_columns]
X_val = df_val[X_columns]
X_test = df_test[X_columns]
model_LR_log.fit(X_train, y_train_log)
y_pred_log = model_LR_log.predict(X_val)
rmse_LR_log = np.sqrt(mean_squared_error(y_val_log, y_pred_log))
rmse_LR_log_exp = np.exp(rmse_LR_log)

# plot the predictions that resulted from linear regression on the log-transformed values;
# to be comparable with the prediction on the non-log-transformed values, take the exponent
y_pred_LR_log = np.exp(y_pred_log)

plt.figure()
sns.histplot (y_pred_LR_log, color = 'red', alpha = 0.5, bins = 50)
sns.histplot(y_val, color = 'blue', alpha = 0.5, bins = 50)
plt.xlim(0, 500)
plt.show()

# logtransform really means improvement in rmse and in the character of data distribution; 
# however, it looks like it was safe to predict low values because of the structure of the dataset,
# so the linear regression got pessimistic

# let's try with polynomial regression on log-transformed data
# we train the model with several order of the polynomials and compare the results
order_polynomial = [2, 3, 4]

for i in order_polynomial: 
    polynomial_features = PolynomialFeatures(degree=i)
    X_train_poly = polynomial_features.fit_transform(X_train)
    X_val_poly = polynomial_features.fit_transform(X_val)
    
    model_polynomial = LinearRegression()
    model_polynomial.fit(X_train_poly, y_train)
    y_poly_pred = model_polynomial.predict(X_val_poly)
    plt.figure()
    sns.histplot (y_poly_pred, color = 'red', alpha = 0.5, bins = 50)
    sns.histplot(y_val, color = 'blue', alpha = 0.5, bins = 50)
    # plt.xlim(0, 500)
    plt.show()
    rmse = np.sqrt(mean_squared_error(y_val, y_poly_pred))
    r2 = r2_score(y_val, y_poly_pred)
    print('polynomial of order', i, 'rmse', rmse)
    
# I choose the 4th order, although it seems it's overfitting, it reproduces the shape of the data distribution best
# besides, both 2nd and 3rd order polynomials ended up predicting some of the Cu values as negative
# does it make sense to work with polynomials of the log-transformed values? 

# Now let's try to use dictvectorizer to code the output variable and predict rock types from chemistry
# first additional manual data clean-up: removing the spaces, checking the rock types for typos
# the dataset is not balanced, but we will see how it gets split;
# an idea is to use only rock types with more than 50 samples in the group

# Logistic regression

from sklearn.linear_model import LogisticRegression
numerical = ['SiO2_pct',
         'TiO2_pct',
         'Al2O3_pct',
         'Fe2O3T_pct',
         'MgO_pct',
         'Sc_ppm',
         'Cr_ppm',
         'MnO_pct',
         'CaO_pct',
         'Na2O_pct',
         'K2O_pct',
         'P2O5_pct',
         'SO3_pct',
         'Sr_ppm',
         'Zr_ppm',
         'Y_ppm',
         'V_ppm',
         'Hf_ppm',
         'Ba_ppm',
         'Nb_ppm']

# choose rock types as y variable

categorical = ['Preferred chemical classification']

# make a new split of the training, validation and test sets trying to keep it balanced with stratify
df_full_train_rocks, df_test_rocks = train_test_split(df_Cu, test_size = 0.2, random_state = 1, stratify=df_Cu[categorical])

# convert objects to numerical
# we still have "objects" instead of numeric values in the dataset, I have no clue why
# so I convert them here explicitly

df_full_train['Cu_ppm'] = df_full_train['Cu_ppm'].astype(float)
df_full_train['Sc_ppm'] = df_full_train['Sc_ppm'].astype(float)
df_full_train['Cr_ppm'] = df_full_train['Cr_ppm'].astype(float)
df_full_train['MnO_pct'] = df_full_train['MnO_pct'].astype(float)
df_full_train['CaO_pct'] = df_full_train['CaO_pct'].astype(float)
df_full_train['Na2O_pct'] = df_full_train['Na2O_pct'].astype(float)
df_full_train['K2O_pct'] = df_full_train['K2O_pct'].astype(float)
df_full_train['SO3_pct'] = df_full_train['SO3_pct'].astype(float)
df_full_train['Sr_ppm'] = df_full_train['Sr_ppm'].astype(float)
df_full_train['Zr_ppm'] = df_full_train['Zr_ppm'].astype(float)
df_full_train['V_ppm'] = df_full_train['V_ppm'].astype(float)
df_full_train['Hf_ppm'] = df_full_train['Hf_ppm'].astype(float)
df_full_train['Ba_ppm'] = df_full_train['Ba_ppm'].astype(float)
df_full_train['Nb_ppm'] = df_full_train['Nb_ppm'].astype(float)

df_train_rocks, df_val_rocks = train_test_split(df_full_train_rocks, test_size = 0.25, stratify=df_full_train_rocks[categorical])

# reset indices 
df_train_rocks = df_train_rocks.reset_index(drop = True)
df_val_rocks = df_train_rocks.reset_index(drop = True)
df_test_rocks = df_train_rocks.reset_index(drop = True)

X_train_rocks = df_train_rocks[numerical].values
X_val_rocks = df_val_rocks[numerical].values
X_test_rocks = df_test_rocks[numerical].values

y_train_rocks = df_train_rocks[categorical].values
y_val_rocks = df_val_rocks[categorical].values
y_test_rocks = df_test_rocks[categorical].values

lr = LogisticRegression()

lr.fit(X_train_rocks, y_train_rocks)

y_pred_rocks = lr.predict(X_val_rocks)
accuracy_LR = (y_pred_rocks == y_val_rocks).mean()

# This was done on the dataset with labels "as is", and whoops, we can only guess the rock type in 10% of the cases. A random number generator could do a better job. 
# Let's take a smaller subset which is balanced

# increase of accuracy to 12% - where did it all go wrong? 

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth = 4)
dt.fit(X_train_rocks, y_train_rocks)
dt_predict = dt.predict(X_val_rocks)

# confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_val_rocks, dt_predict)
accuracy_dt = (dt_predict == y_val_rocks).mean()

rocks_in_train = np.unique(df_train_rocks[categorical].values)
rocks_in_val = np.unique(df_val_rocks[categorical].values)
rocks_in_test = np.unique(df_test_rocks[categorical].values) 
   
# save the confusion matrix
conf = confusion_matrix(y_val_rocks, dt_predict)
df = pd.DataFrame(conf)

plt.figure(figsize = (15,15))
cm = conf
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, cmap='Greens'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix, decision trees'); 
ax.xaxis.set_ticklabels(['Basalt High-Ti', 'Dolerite LTB', 'Dolerite/gabbro LTB', 'EMI',
       'Felsic dyke (dacitic)', 'Felsic dyke (rhyolitic)',
       'Felsic vol/volclas (andesitic)', 'Felsic vol/volclas (dacitic)',
       'Felsic vol/volclas (rhyolitic)', 'Granite', 'HTSB', 'ITB',
       'Komatiite', 'Komatiite High-Al', 'LTB', 'LTB High-Nb',
       'Problematic data'], rotation = 90); 
ax.yaxis.set_ticklabels(['Basalt High-Ti', 'Dolerite LTB', 'Dolerite/gabbro LTB', 'EMI',
       'Felsic dyke (dacitic)', 'Felsic dyke (rhyolitic)',
       'Felsic vol/volclas (andesitic)', 'Felsic vol/volclas (dacitic)',
       'Felsic vol/volclas (rhyolitic)', 'Granite', 'HTSB', 'ITB',
       'Komatiite', 'Komatiite High-Al', 'LTB', 'LTB High-Nb',
       'Problematic data'], rotation = 45);

plt.savefig('confusion matrix dtree.jpg', dpi = 300) 

# I have 17 groups! Why are there 20?  
res = pd.DataFrame(dt_predict)
res.to_csv("prediction_results.csv")

# when in doubt, use Gradient Boost

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train_rocks, y_train_rocks)
gb.fit(X_train_rocks, y_train_rocks)
gb_predict = gb.predict(X_val_rocks)
gb_score = gb.score(X_val_rocks, y_val_rocks)

confusion_matrix(y_val_rocks, gb_predict)
accuracy_gb = (gb_predict == y_val_rocks).mean()


# save the confusion matrix
conf_gb = confusion_matrix(y_val_rocks, gb_predict)
df = pd.DataFrame(conf_gb)

plt.figure(figsize = (15,15))
cm = conf_gb
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, cmap='Greens'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix, Gradient boost'); 
ax.xaxis.set_ticklabels(['Basalt High-Ti', 'Dolerite LTB', 'Dolerite/gabbro LTB', 'EMI',
       'Felsic dyke (dacitic)', 'Felsic dyke (rhyolitic)',
       'Felsic vol/volclas (andesitic)', 'Felsic vol/volclas (dacitic)',
       'Felsic vol/volclas (rhyolitic)', 'Granite', 'HTSB', 'ITB',
       'Komatiite', 'Komatiite High-Al', 'LTB', 'LTB High-Nb',
       'Problematic data'], rotation = 90); 
ax.yaxis.set_ticklabels(['Basalt High-Ti', 'Dolerite LTB', 'Dolerite/gabbro LTB', 'EMI',
       'Felsic dyke (dacitic)', 'Felsic dyke (rhyolitic)',
       'Felsic vol/volclas (andesitic)', 'Felsic vol/volclas (dacitic)',
       'Felsic vol/volclas (rhyolitic)', 'Granite', 'HTSB', 'ITB',
       'Komatiite', 'Komatiite High-Al', 'LTB', 'LTB High-Nb',
       'Problematic data'], rotation = 45);

plt.savefig('confusion matrix gboost.jpg', dpi = 300) 

# whatever. Random forest. 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
n_estimators = [50,100,200,500,1000]
max_depth = [3,5,10,15,20,None]
##
parameters = {'n_estimators':n_estimators, 'max_depth':max_depth}
classifier = RandomForestClassifier(random_state=27)
modelGS = GridSearchCV(estimator=classifier,param_grid=parameters,cv=10,n_jobs=-1)
modelGS.fit(X_train_rocks, y_train_rocks)

modelGS_predict = modelGS.predict(X_val_rocks)
IGS_score = modelGS.score(X_val_rocks, y_val_rocks)

confusion_matrix(y_val_rocks, modelGS_predict) 
accuracy_rf = (modelGS_predict == y_val_rocks).mean()


# save the confusion matrix
conf_rf = confusion_matrix(y_val_rocks, modelGS_predict)
df = pd.DataFrame(conf_rf)

plt.figure(figsize = (15,15))
cm = conf_rf
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, cmap='Greens'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix, random forest'); 
ax.xaxis.set_ticklabels(['Basalt High-Ti', 'Dolerite LTB', 'Dolerite/gabbro LTB', 'EMI',
       'Felsic dyke (dacitic)', 'Felsic dyke (rhyolitic)',
       'Felsic vol/volclas (andesitic)', 'Felsic vol/volclas (dacitic)',
       'Felsic vol/volclas (rhyolitic)', 'Granite', 'HTSB', 'ITB',
       'Komatiite', 'Komatiite High-Al', 'LTB', 'LTB High-Nb',
       'Problematic data'], rotation = 90); 
ax.yaxis.set_ticklabels(['Basalt High-Ti', 'Dolerite LTB', 'Dolerite/gabbro LTB', 'EMI',
       'Felsic dyke (dacitic)', 'Felsic dyke (rhyolitic)',
       'Felsic vol/volclas (andesitic)', 'Felsic vol/volclas (dacitic)',
       'Felsic vol/volclas (rhyolitic)', 'Granite', 'HTSB', 'ITB',
       'Komatiite', 'Komatiite High-Al', 'LTB', 'LTB High-Nb',
       'Problematic data'], rotation = 45);

plt.savefig('confusion random forest.jpg', dpi = 300) 

# let's see how it performs on the test set

dt_test_predict = dt.predict(X_test_rocks)
# save the confusion matrix
conf_dt = confusion_matrix(y_test_rocks, dt_test_predict)
df = pd.DataFrame(conf_dt)

plt.figure(figsize = (15,15))
cm = conf_dt
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, cmap='Greens'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix, decision trees, test set'); 
ax.xaxis.set_ticklabels(['Basalt High-Ti', 'Dolerite LTB', 'Dolerite/gabbro LTB', 'EMI',
       'Felsic dyke (dacitic)', 'Felsic dyke (rhyolitic)',
       'Felsic vol/volclas (andesitic)', 'Felsic vol/volclas (dacitic)',
       'Felsic vol/volclas (rhyolitic)', 'Granite', 'HTSB', 'ITB',
       'Komatiite', 'Komatiite High-Al', 'LTB', 'LTB High-Nb',
       'Problematic data'], rotation = 90); 
ax.yaxis.set_ticklabels(['Basalt High-Ti', 'Dolerite LTB', 'Dolerite/gabbro LTB', 'EMI',
       'Felsic dyke (dacitic)', 'Felsic dyke (rhyolitic)',
       'Felsic vol/volclas (andesitic)', 'Felsic vol/volclas (dacitic)',
       'Felsic vol/volclas (rhyolitic)', 'Granite', 'HTSB', 'ITB',
       'Komatiite', 'Komatiite High-Al', 'LTB', 'LTB High-Nb',
       'Problematic data'], rotation = 45);

plt.savefig('confusion matrix dtree test.jpg', dpi = 300) 

gb_predict_test = gb.predict(X_test_rocks)
gb_score = gb.score(X_test_rocks, y_test_rocks)

confusion_matrix(y_test_rocks, gb_predict)
accuracy_gb = (gb_predict_test == y_test_rocks).mean()


# save the confusion matrix
conf_gb_test = confusion_matrix(y_test_rocks, gb_predict_test)
df = pd.DataFrame(conf_gb_test)

plt.figure(figsize = (15,15))
cm = conf_gb_test
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, cmap='Greens'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix, Gradient boost, test set'); 
ax.xaxis.set_ticklabels(['Basalt High-Ti', 'Dolerite LTB', 'Dolerite/gabbro LTB', 'EMI',
       'Felsic dyke (dacitic)', 'Felsic dyke (rhyolitic)',
       'Felsic vol/volclas (andesitic)', 'Felsic vol/volclas (dacitic)',
       'Felsic vol/volclas (rhyolitic)', 'Granite', 'HTSB', 'ITB',
       'Komatiite', 'Komatiite High-Al', 'LTB', 'LTB High-Nb',
       'Problematic data'], rotation = 90); 
ax.yaxis.set_ticklabels(['Basalt High-Ti', 'Dolerite LTB', 'Dolerite/gabbro LTB', 'EMI',
       'Felsic dyke (dacitic)', 'Felsic dyke (rhyolitic)',
       'Felsic vol/volclas (andesitic)', 'Felsic vol/volclas (dacitic)',
       'Felsic vol/volclas (rhyolitic)', 'Granite', 'HTSB', 'ITB',
       'Komatiite', 'Komatiite High-Al', 'LTB', 'LTB High-Nb',
       'Problematic data'], rotation = 45);

plt.savefig('confusion matrix gboost test.jpg', dpi = 300) 

modelGS_predict_test = modelGS.predict(X_test_rocks)
IGS_score_test = modelGS.score(X_test_rocks, y_test_rocks)

confusion_matrix(y_test_rocks, modelGS_predict) 
accuracy_rf = (modelGS_predict == y_test_rocks).mean()


# save the confusion matrix
conf_rf_test = confusion_matrix(y_test_rocks, modelGS_predict_test)
df = pd.DataFrame(conf_rf_test)

plt.figure(figsize = (15,15))
cm = conf_rf_test
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, cmap='Greens'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix, random forest, test set'); 
ax.xaxis.set_ticklabels(['Basalt High-Ti', 'Dolerite LTB', 'Dolerite/gabbro LTB', 'EMI',
       'Felsic dyke (dacitic)', 'Felsic dyke (rhyolitic)',
       'Felsic vol/volclas (andesitic)', 'Felsic vol/volclas (dacitic)',
       'Felsic vol/volclas (rhyolitic)', 'Granite', 'HTSB', 'ITB',
       'Komatiite', 'Komatiite High-Al', 'LTB', 'LTB High-Nb',
       'Problematic data'], rotation = 90); 
ax.yaxis.set_ticklabels(['Basalt High-Ti', 'Dolerite LTB', 'Dolerite/gabbro LTB', 'EMI',
       'Felsic dyke (dacitic)', 'Felsic dyke (rhyolitic)',
       'Felsic vol/volclas (andesitic)', 'Felsic vol/volclas (dacitic)',
       'Felsic vol/volclas (rhyolitic)', 'Granite', 'HTSB', 'ITB',
       'Komatiite', 'Komatiite High-Al', 'LTB', 'LTB High-Nb',
       'Problematic data'], rotation = 45);

plt.savefig('confusion random forest test set.jpg', dpi = 300) 