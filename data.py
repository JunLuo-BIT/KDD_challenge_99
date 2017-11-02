#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:23:39 2017

@author: austin
"""
# cd /home/austin/ML/KDDcup99
import pandas
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss

# Must declare data_dir as the directory of training and test files
data_dir = '/home/austin/ML/KDDcup99/'

train_data = data_dir + "kddcup.data_10_percent_corrected"
train_labels = data_dir + "train_labels.txt"
test_data = data_dir + "correctedk"
test_labels = data_dir + "test_labels.txt"

#data['malware'] = data['malware'].apply(lambda Tag: 0 if Tag=='normal.' else 1 )


def process_data(X, y):
    X = X.drop(41, 1)
    X[1], uniques = pandas.factorize(X[1])
    X[2], uniques = pandas.factorize(X[2])
    X[3], uniques = pandas.factorize(X[3])

    num_examples = 10**6
    X = X[0:num_examples]
    y = y[0:num_examples]

    X = numpy.array(X)
    y = numpy.array(y).ravel()

    return X, y

data = pandas.read_csv(train_data,header=None)

print("Loading training data")
train_X = pandas.read_csv(train_data, header=None)
train_y = pandas.read_csv(train_labels, header=None)
train_X, train_y = process_data(train_X, train_y)

print("Loading test data")
test_X = pandas.read_csv(test_data, header=None)
test_y = pandas.read_csv(test_labels, header=None)
test_X, test_y = process_data(test_X, test_y)

print("Training and predicting")
learner = KNeighborsClassifier(1, n_jobs=-1)
learner.fit(train_X, train_y)
pred_y = learner.predict(test_X)

results = confusion_matrix(test_y, pred_y)
error = zero_one_loss(test_y, pred_y)

print(results)
print(error)