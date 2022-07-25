# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:31:05 2022

@author: intan
"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,train_test_split
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer,IterativeImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os

#%%

def cramers_corrected_stat(conf_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(conf_matrix)[0]
    n = conf_matrix.sum()
    phi2 = chi2/n
    r,k = conf_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
#%%

CSV_PATH = os.path.join(os.getcwd(),'dataset','heart.csv')

#%% Step 1) Data Loading

df = pd.read_csv(CSV_PATH)

#%% Step 2)2)Data inspection/visualization
df.describe().T # to obtain basic stats
df.info() # to tell if there is any NANs or the data type

#continuous
con_col=['age','trtbps','chol','thalachh','oldpeak']

for i in con_col:
    plt.figure()
    sns.distplot(df[i])
    plt.show()

#categorical
cat_col=df.drop(labels=con_col,axis=1).columns

for i in cat_col:
    plt.figure()
    sns.countplot(df[i])
    plt.show()

df.boxplot(figsize=(11,7))



#trtbps and chol have outliers

#%%3) Data cleaning
#1) Outliers
#The outlier is still within the range
#2) NANs

df['thall']=df['thall'].replace(0,np.nan) #replacing out of range value to NaN
df['caa']=df['caa'].replace(4,np.nan) #replacing out of range value to NaN
df.isna().sum() #total of 7 NANs

df['thall']=df['thall'].fillna(df['thall'].mode()[0])
df['caa']=df['caa'].fillna(df['caa'].mode()[0])

df.isna().sum()

#3) Duplicated
df.duplicated().sum() #there's 1 duplicate
df=df.drop_duplicates() #drop duplicates

#%% Step 4) Features selection

selected_features=[]

for i in con_col:
    print(i)
    lr=LogisticRegression()
    lr.fit(np.expand_dims(df[i],axis=-1),df['output']) #X:cont Y:Cat
    print(lr.score(np.expand_dims(df[i],axis=-1),df['output']))
    if lr.score(np.expand_dims(df[i],axis=-1),df['output'])>0.6:
        selected_features.append(i)
        
for i in cat_col:
    print(i)
    confusion_matrix=pd.crosstab(df[i], df['output']).to_numpy()
    print(cramers_corrected_stat(confusion_matrix))
    if cramers_corrected_stat(confusion_matrix)> 0.4:
        selected_features.append(i)
 
print(selected_features)

#%% Step 5) Data preprocessing

df=df.loc[:,selected_features]
X=df.drop(labels='output',axis=1)
y=df['output']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,
                                               random_state=123)

#%% Data Modelling

pipeline_mms_lr = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('Logistic_Classifier',LogisticRegression())
                            ]) #Pipeline([STEPS])

pipeline_ss_lr = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('Logistic_Classifier',LogisticRegression())
                            ]) #Pipeline([STEPS])

pipeline_mms_dt = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('Tree_Classifier',DecisionTreeClassifier())
                            ]) #Pipeline([STEPS])

pipeline_ss_dt = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('Tree_Classifier',DecisionTreeClassifier())
                            ]) #Pipeline([STEPS])

pipeline_mms_rf = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('rf_Classifier',RandomForestClassifier())
                            ]) #Pipeline([STEPS])

pipeline_ss_rf = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('rf_Classifier',RandomForestClassifier())
                            ]) #Pipeline([STEPS])


pipeline_mms_knn = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('knn_Classifier',KNeighborsClassifier(n_neighbors=3))
                            ]) #Pipeline([STEPS])

pipeline_ss_knn = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('knn_Classifier',KNeighborsClassifier(n_neighbors=3))
                            ]) #Pipeline([STEPS])

pipeline_mms_gb = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('gb_Classifier',GradientBoostingClassifier())
                            ]) #Pipeline([STEPS])

pipeline_ss_gb = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('gb_Classifier',GradientBoostingClassifier())
                            ]) #Pipeline([STEPS])

pipeline_mms_svc = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('SVC_Classifier',SVC())
                            ]) #Pipeline([STEPS])

pipeline_ss_svc = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('SVC_Classifier',SVC())
                            ]) #Pipeline([STEPS])


# create a list to store all the pipelines
pipelines = [pipeline_mms_lr, pipeline_ss_lr, pipeline_mms_dt,pipeline_ss_dt,
             pipeline_mms_knn,pipeline_ss_knn, pipeline_mms_gb, pipeline_ss_gb,
             pipeline_mms_svc,pipeline_ss_svc]

for pipe in pipelines:
    pipe.fit(X_train, y_train)

best_accuracy= 0

for i,pipe in enumerate(pipelines):
    print(pipe.score(X_test,y_test))
    if pipe.score(X_test, y_test) > best_accuracy:
        best_accuracy=pipe.score(X_test, y_test)
        best_pipeline = pipe

print('The best scaler and classifier for diabetes data is {} with accuracy of {}'.
      format(best_pipeline.steps, best_accuracy))

#%%

pipeline_mms_lr = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('Logistic_Classifier',LogisticRegression())
                            ]) #Pipeline([STEPS])

grid_param=[{'Logistic_Classifier__C':np.arange(1,5,.1),
             'Logistic_Classifier__intercept_scaling':np.arange(1,10,1),
             'Logistic_Classifier__solver':['newton-cg','lbfgs','liblinear','sag','saga']}]

gridsearch=GridSearchCV(pipeline_mms_lr, grid_param,cv=5,verbose=1,n_jobs=-1)
grid=gridsearch.fit(X_train,y_train)
grid.score(X_test,y_test)
print(grid.best_score_)
print(grid.best_params_)

best_model=grid.best_estimator_
#%%Model evaluation
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,confusion_matrix
y_pred=best_model.predict(X_test)
print(classification_report(y_test, y_pred))

cm=confusion_matrix(y_test,y_pred)

labels=['less chance','risky']
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()
#%% Model Saving

BEST_ESTIMATOR_SAVE_PATH=os.path.join(os.getcwd(),'model','best_estimator.pkl')

with open(BEST_ESTIMATOR_SAVE_PATH, 'wb') as file:
    pickle.dump(best_model,file)