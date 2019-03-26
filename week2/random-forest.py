import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
import sklearn.metrics
 # Feature Importance
from sklearn.ensemble import ExtraTreesClassifier

os.chdir("E:\Python_Course")

#Load the dataset

AH_data = pd.read_csv("gap.csv")
data_clean = AH_data.fillna(AH_data.mean())

data_clean.dtypes
data_clean.describe()

#Split into training and testing sets

predictors = data_clean[['incomeperperson','alcconsumption','armedforcesrate',
                         'breastcancerper100th','co2emissions','femaleemployrate',
                         'hivrate','lifeexpectancy','oilperperson',
                         'polityscore','relectricperperson','suicideper100th','employrate','urbanrate']]

targets = data_clean.target

pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, targets, test_size=.4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)

# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)
# display the relative importance of each attribute
print(model.feature_importances_)


"""
Running a different number of trees and see the effect
 of that on the accuracy of the prediction
"""

trees=range(25)
accuracy=np.zeros(25)

for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(pred_train,tar_train)
   predictions=classifier.predict(pred_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)
   
plt.cla()
plt.plot(trees, accuracy)