# ML zoomcamp midterm project: supervised learning for Archaean rocks; 
based on Scott Halley’s geochemistry tutorial “Eastern Goldfields Greenstone Geochemical Barcoding Project” 

There is a great tutorial on interpreting this dataset that I would try to build on https://www.scotthalley.com.au/tutorials (Lithogeochemistry of Archean Rocks)

https://dmpbookshop.eruditetechnologies.com.au/product/eastern-goldfields-greenstone-geochemical-barcoding-project-notes-to-accompany-2021-data-release.do

The dataset contains geology descriptions and chemical data for 5447 rock samples from the Eastern Goldfields region (Western Australia), around 100 features (elements, coordinates, rock types and descriptions), of which I would take a few important numerical features: 
Si, Ti, Al, Fe, Mg, Mn, Ca, Na, K, P, S, V - major and minor rock-forming elements
Sc, Sr, Zr, Y, Hf, Ba, Nb - trace elements that could be important for predicting magma fertility. 

# Purpose of the project
To investigate the dependencies between different elements and see if we can predict target variables like rock types and ore elements from the other elements
using regression, decision trees and random forest; for the second part of the course – use 1d convolution and unsupervised learning (clustering and PCA)

# Data clean-up
Replacing the values below the detection limit with half of the detection limit, looking at the lithology types, merging the labels whenever possible and if only a few samples were present in the group, replacing the values with an extra space by identical values without an extra space, changing EIM-like to EMI-like, merging all versions of “komatiite high-Al, komatiite High Al etc”. Like always in geology, the dataset is not balanced when it comes to rock types, but we have to live with it for the moment. 

# Feature selection
After checking the correlation of the selected elements to the target variable, I left all the primary elements as input. 

# EDA 
The target "Cu" variable has a long tail, hence I will use log transformation before applying regression. 

# Linear and polynomial regression on the validation set
After applying log-transform and plotting the predicted Cu values on top of actual Cu values from the validation dataset, we see that the result is not too bad for the 4th order polynomial, and that linear regression wants to be pessimistic and predicts low Cu values. Predictions are in red, actual values are in blue. 

<img src=https://github.com/DinaKlim/Alexey-Grigoriev-s-ML-Zoomcamp-Homework/blob/main/Midterm%20project/Linear%20regression.png>

Here is linear regression being pessimistic about Cu content for the validation dataset

<img src=https://github.com/DinaKlim/Alexey-Grigoriev-s-ML-Zoomcamp-Homework/blob/main/Midterm%20project/4th%20order%20polynomial.png>

and here is the polynomial regression, 4th order, performing ok on the validation dataset with the rmse of 2.6 ppm (which means that we will be 2.6 ppm off when we try to predict Cu content). 

# Basic plots for interpreting mafic and ultramafic rocks: work in progress
These plots reproduce the beginning of Scott Halley's tutorial; here I am just getting started.
<img src=https://github.com/DinaKlim/Alexey-Grigoriev-s-ML-Zoomcamp-Homework/blob/main/Midterm%20project/chemistry1.png>
<img src=https://github.com/DinaKlim/Alexey-Grigoriev-s-ML-Zoomcamp-Homework/blob/main/Midterm%20project/chemistry2.png>
Basic chemistry plots based on Morgan Williams' pyrolite library

<img src=https://github.com/DinaKlim/Alexey-Grigoriev-s-ML-Zoomcamp-Homework/blob/main/Midterm%20project/Jensen%20plot%20attempt.png>
(this Jensen plot really needs improvement, but I failed to do a density plot and draw the lines - comments and corrections would be appeciated!)

# Predicting rock types from geochemistry
As some of the rock types were represented by only one sample, I have selected a subset of rock groups that contained more than 100 samples, which included: 
- Basalt High-Ti
- Dolerite LTB
- Dolerite/gabbro LTB
- EMI
- Felsic dyke (dacitic)
- Felsic dyke (rhyolitic)
- Felsic vol/volclas (andesitic)
- Felsic vol/volclas (dacitic)
- Felsic vol/volclas (rhyolitic)
- Granite
- HTSB
- ITB
- Komatiite
- Komatiite High-Al
- LTB
- LTB High-Nb
- Problematic data

These 17 groups contain 4461 samples out of 5446 (81 groups). In order to maintain the data structure, stratify was used when splitting the data into training, validation and test sets. 

Decision trees, gradientboost and random forest produced very different results for the validation dataset. 

I use confusion matrix to try to understand how good the predictions are; predicted labels are on a horizontal axis, true labels are on a vertical axis, and the number in the cell shows how many samples were classified correctly (when they sit on a diagonal, i.e. predicted labels coincide with true labels), and how many were misclassified and in what way. 
When plotting the confusion matrix, I got a mismatch of the group numbers! But found that extra space after the EMI rock label, that was sometimes there, sometimes not, sometimes there were two of them. Got the unique labels with np.unique, got back to the dataset, cleaned the extra spaces. 
<img src=https://github.com/DinaKlim/Alexey-Grigoriev-s-ML-Zoomcamp-Homework/blob/main/Midterm%20project/confusion%20matrix%20dtree%20test.jpg>
Decision tree predictions on the test set: let's talk about what gets confused and why? 
<img src=https://github.com/DinaKlim/Alexey-Grigoriev-s-ML-Zoomcamp-Homework/blob/main/Midterm%20project/confusion%20matrix%20gboost%20test.jpg>
Gradient boost is really confused? 
<img src=https://github.com/DinaKlim/Alexey-Grigoriev-s-ML-Zoomcamp-Homework/blob/main/Midterm%20project/confusion%20random%20forest%20test%20set.jpg>
... and a classic overfitting example with random forest. 

# Conclusion and to follow-up
It is probably possible to predict the Cu variable relatively well for the validation set, but in order to make a prediction we would need to split the dataset based on location and see if we can predict Cu for a neighboring area based on a similar area.  
There is enough food for thought in Scott Halley's tutorial to last a lifetime, and I have not written up on any geological reasoning behind the geochemistry here; that's on  to-do list. 
Sorry Alexey, I know you encouraged to stop training and do deployment, but I can't stop training and due to lack of programming experience the pickle/flask is beyond me for the moment; another item in the to-do list. 
Good validation score is missing: ran out of time. 
