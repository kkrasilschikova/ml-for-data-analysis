# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics

os.chdir("E:\Python_Course")

#Load the dataset
AH_data = pd.read_csv("gap.csv")

data_clean = AH_data.dropna()
#data_clean = AH_data.fillna(AH_data.mean()) -- this is the second run

data_clean.dtypes
data_clean.describe()

#Split into training and testing sets
predictors = data_clean[['incomeperperson','alcconsumption','armedforcesrate',
                         'breastcancerper100th','co2emissions','femaleemployrate',
                         'hivrate','lifeexpectancy','oilperperson',
                         'polityscore','relectricperperson','suicideper100th','employrate','urbanrate']]

targets = data_clean.target

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size=.4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)

#Displaying the decision tree
from sklearn import tree
from io import StringIO
out = StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
graph.write_png('MyGap.png')
#graph.write_png('MyGapMean.png') -- this is the second run
