#from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV

os.chdir("E:\Python_Course")
 
#Load the dataset
data = pd.read_csv("gap.csv")

# Data Management
data_clean = data.fillna(data.mean())

# Get column names first
cols = data_clean[['incomeperperson','alcconsumption','armedforcesrate','breastcancerper100th',
                   'co2emissions','femaleemployrate','hivrate','lifeexpectancy','oilperperson',
                   'polityscore','relectricperperson','suicideper100th','employrate','urbanrate','target']]

# Fit data on the scaler object
names = cols.columns
from sklearn import preprocessing

# Create the Scaler object
scaler = preprocessing.MinMaxScaler()
scaled = scaler.fit_transform(cols)
scaled = pd.DataFrame(scaled, columns=names)

#select predictor variables and target variable as separate data sets  
predvar = scaled[['incomeperperson','alcconsumption','armedforcesrate','breastcancerper100th',
                  'co2emissions','femaleemployrate','hivrate','lifeexpectancy','oilperperson',
                  'polityscore','relectricperperson','suicideper100th','employrate','urbanrate']]

target = scaled.target

# standardize predictors to have mean=0 and sd=1
predictors=predvar.copy()
predictors['incomeperperson']=preprocessing.scale(predictors['incomeperperson'].astype('float64'))
predictors['alcconsumption']=preprocessing.scale(predictors['alcconsumption'].astype('float64'))
predictors['armedforcesrate']=preprocessing.scale(predictors['armedforcesrate'].astype('float64'))
predictors['breastcancerper100th']=preprocessing.scale(predictors['breastcancerper100th'].astype('float64'))
predictors['co2emissions']=preprocessing.scale(predictors['co2emissions'].astype('float64'))
predictors['femaleemployrate']=preprocessing.scale(predictors['femaleemployrate'].astype('float64'))
predictors['hivrate']=preprocessing.scale(predictors['hivrate'].astype('float64'))
predictors['lifeexpectancy']=preprocessing.scale(predictors['lifeexpectancy'].astype('float64'))
predictors['oilperperson']=preprocessing.scale(predictors['oilperperson'].astype('float64'))
predictors['polityscore']=preprocessing.scale(predictors['polityscore'].astype('float64'))
predictors['relectricperperson']=preprocessing.scale(predictors['relectricperperson'].astype('float64'))
predictors['suicideper100th']=preprocessing.scale(predictors['suicideper100th'].astype('float64'))
predictors['employrate']=preprocessing.scale(predictors['employrate'].astype('float64'))
predictors['urbanrate']=preprocessing.scale(predictors['urbanrate'].astype('float64'))

# split data into train and test sets
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, 
                                                              test_size=.4, random_state=123)

# specify the lasso regression model
model=LassoLarsCV(cv=5, precompute=False).fit(pred_train,tar_train)

# print variable names and regression coefficients
dict(zip(predictors.columns, model.coef_))

# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')

# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.mse_path_, ':')
plt.plot(m_log_alphascv, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
         

# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data mean_squared_error')
print(train_error)
print ('test data data mean_squared_error')
print(test_error)

# R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)