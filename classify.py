#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:23:39 2017

@author: austin
"""
# cd /home/austin/ML/KDDcup99
# cd /Users/spdllab02/Documents/ADML/KDD_challenge_99

import pandas as pd
import numpy as np 

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

import time

def datapreprocessing(data):
    data['malware'] = data['malware'].apply(lambda Tag: 0 if Tag=='normal.' else 1 )
    #Encoding the categorical values into numerical values.
    le = LabelEncoder()
    data.protocol_type = le.fit_transform(data.protocol_type)
    data.service = le.fit_transform( data.service)
    data.flag = le.fit_transform( data.flag)
    # Standard way of dealing with categorical values
    #   -  use label encoder to convert categorical into number.
    #   -  Use OneHotEncoder to convert numbers to binary.
    
    normal_df = (data.loc[data['malware'] == 0]).sample(n=50000)
    attack_df = (data.loc[data['malware'] == 1]).sample(n=50000)
    data = normal_df.append(attack_df)
    
    X=data.iloc[:,:41].as_matrix().astype(np.float)
    y=data.iloc[:,-1].as_matrix().astype(np.float)
    return X,y

def featureSelect(data):
    features=list(data)
    data.drop('land',axis=1)
    data.drop('wrng_fragment',axis=1)
    data.drop('urgent',axis=1)
    data.drop('su_atempted',axis=1)
    data.drop('num_root',axis=1)
    data.drop('num_accesed_files',axis=1)
    data.drop('is_host_login',axis=1)
    data.drop('is_guest_login',axis=1)
    data.drop('count',axis=1)
    data.drop('root_shell',axis=1)
    data.drop('num_file_creations',axis=1)
    data.drop('num_shells',axis=1)
    data.drop('num_outbound_cmds',axis=1)
    data.drop('num_failed_flog_in',axis=1)
    return data

def stratified_cv(X, y, clf_class, clf_name, shuffle=True, n_folds=2, **kwargs):
    stratified_k_fold = StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle) #it will have each folds with index of y. Use this index to get corresponding x values 
    y_pred = y.copy()
    clf = clf_class(**kwargs)
    
    start_time=time.time()
    #Iterate throught the folds. we will have ii part with 90% of train data index and jj part with 10% of test data index.
    for ii, jj in stratified_k_fold:
        X_train, X_test = X[ii], X[jj]
        y_train = y[ii]
        clf.fit(X_train,y_train)
        y_pred[jj] = clf.predict(X_test)
    accuracy = accuracy_score(y_pred,y)*100
    print(str(clf_name)+"\t" +"%.3f"%(time.time()-start_time)+"\t"+ str(accuracy))
    return {'name':clf_name,'y_pred':y_pred ,'acc':accuracy}

         # To get value distribution of features
#for x in list(data)[30:]:
#    print(x,'\t',len(set(data[x])) )
#    for i in set(data[x]):
#        print(i, len(data.loc[data[x]==i]))
#    print("\n")

#def main():
data = pd.read_csv('kddcup.data_10_percent_corrected')  
data=data.drop_duplicates()
data=featureSelect(data)
X,y=datapreprocessing(data)

print("Classifier\t Execution Time \t Accuracy")
#RFC = stratified_cv(X, y, RandomForestClassifier, "Random Forest Classifier",max_features=30 )
KNN = stratified_cv(X,y, KNeighborsClassifier, "K Neighbor Classifier", n_neighbors=1000)
SVMC = stratified_cv(X,y, SVC, "Support Vector Machine",max_iter=300)

#
#if __name__ == '__main__':
#    main()