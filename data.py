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

x=data.applymap(np.isreal)
data['malware'] = data['malware'].apply(lambda Tag: 0 if Tag=='normal.' else 1 )


#Encoding the categorical values into numerical values.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data.protocol_type = le.fit_transform(data.protocol_type)
data.service = le.fit_transform( data.service)
data.flag = le.fit_transform( data.flag)


# Standard way of dealing with categorical values
#   -  use label encoder to convert categorical into number.
#   -  Use OneHotEncoder to convert numbers to binary.

normal_df = (data.loc[data['malware'] == 0]).sample(n=90000)
attack_df = (data.loc[data['malware'] == 1]).sample(n=90000)
data = normal_df.append(attack_df)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(data.iloc[:,:41],data.iloc[:,-1],test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test) 

from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)
