#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print ''
#print features_train[:3]
#print labels_train[:3]
#print features_test[:3]
#print labels_test[:3]
#print ''

#########################################################
### your code goes here ###

from sklearn.naive_bayes import GaussianNB

print 'Create Gaussian Naive Bayes classifier.'
clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

print "accuracy:", round(clf.score(features_test, labels_test), 3)
print "mislabeled:", (clf.predict(features_test)!=labels_test).sum(), "/", len(labels_test)
print ''

#########################################################

from sklearn.naive_bayes import MultinomialNB

print 'Create Naive Bayes classifier for multinomial models.'
clf = MultinomialNB()

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

print "accuracy:", round(clf.score(features_test, labels_test), 3)
print "mislabeled:", (clf.predict(features_test)!=labels_test).sum(), "/", len(labels_test)
print ''

#########################################################

from sklearn.naive_bayes import BernoulliNB

print 'Create Naive Bayes classifier for multivariate Bernoulli models.'
clf = BernoulliNB()

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

print "accuracy:", round(clf.score(features_test, labels_test), 3)
print "mislabeled:", (clf.predict(features_test)!=labels_test).sum(), "/", len(labels_test)
print ''

#########################################################

