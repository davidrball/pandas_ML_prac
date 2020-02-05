#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:33:17 2020

@author: dball
"""

#MNIST classification example
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


#from sklearn.datasets import fetch_mldata
#mnist = fetch_openml('mnist_784',version=1,cache=True)

X,y = mnist["data"],mnist["target"]


#X shape is 70000 by 784
#y shape is 70000

#y shape is just 1d and the size of the number of data points ('targets') we have
#X shape is number of targets by number of features

#i.e., there are 70000 images, and each image has 784 features
some_digit = X[36000]
#some_digit_image = some_digit.reshape(28,28)
#plt.imshow(some_digit_image,interpolation="nearest",cmap="binary")

#looks like a 9, let's check the y val
#print(y[36000]) #yep

#even though it's already divided into training and test, let's do it ourselves for practice

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

#now shuffle, to make sure all cross-validation folds will be similar
trainsize = y_train.size
shuffle_index = np.random.permutation(trainsize)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#need to create target vectors for classification
y_train_9 = (y_train=='9')
y_test_9 = (y_test=='9')

#first we just train a binary classifier, i.e., 5 or 'not' 5
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter=5, tol=-np.Infinity,random_state=42)
sgd_clf.fit(X_train, y_train_9)

#now measure accuracy, writing it ourselves:
'''
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3,random_state=42)

for train_index, test_index in skfolds.split(X_train,y_train_9):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_9[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_9[test_index])
    
    clone_clf.fit(X_train_folds,y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct=sum(y_pred==y_test_fold)
    print(n_correct / len(y_pred))
'''
#or just use the cross val score function
from sklearn.model_selection import cross_val_score
scores = cross_val_score(sgd_clf,X_train,y_train_9,cv=3,scoring="accuracy")

#displaying scores
def display_scores(scores):
    print("Scores : ", scores)
    print("Mean : ", scores.mean())
    print("Standard deviation : ", scores.std())

#display_scores(scores)


#scores based on accuracy not that useful, instead look at confusion matrix
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_9,cv=3)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train_9,y_train_pred))

#if you want more concise performance metric, use recall or precision
from sklearn.metrics import precision_score, recall_score, f1_score
print(precision_score(y_train_9,y_train_pred))
print(recall_score(y_train_9,y_train_pred))
print(f1_score(y_train_9,y_train_pred))

