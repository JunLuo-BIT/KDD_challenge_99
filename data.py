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

data = pd.read_csv('kddcup.data_10_percent_corrected')
data= data.drop_duplicates()
data.reindex()
x=data.applymap(np.isreal)
data['malware'] = data['malware'].apply(lambda Tag: 0 if Tag=='normal.' else 1 )

from sklearn.preprocessing import LabelEncoder
#Encoding the categorical values into numerical values.

le = LabelEncoder()
data.protocol_type = le.fit_transform(data.protocol_type)
data.service = le.fit_transform( data.service)
data.flag = le.fit_transform( data.flag)
data.drop_duplicates()


# Standard way of dealing with categorical values
#   -  use label encoder to convert categorical into number.
#   -  Use OneHotEncoder to convert numbers to binary.

normal_df = (data.loc[data['malware'] == 0]).sample(n=50000)
attack_df = (data.loc[data['malware'] == 1]).sample(n=50000)
data = normal_df.append(attack_df)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = scaler.fit_transform(data.iloc[:,:41])
y=data.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)

import time

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=1,max_features=30 )

start_time = time.time()
rfc.fit(X_train,y_train)
print(time.time()-start_time)

from sklearn import svm
svm_clf = svm.SVC()

start_time = time.time()
svm_clf.fit(X_train,y_train)
print(time.time()-start_time)

svm_pred= svm_clf.predict(X_test)

y_pred = rfc.predict(X_test) 

from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)
accuracy_score(svm_pred,y_test)
