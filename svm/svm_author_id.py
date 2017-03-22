#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
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

"""
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

print ''
print 'Using 1% of training sample...'
print ''
#"""
#########################################################
### your code goes here ###

from sklearn.svm import SVC
"""
print 'Create SVM classfier with linear kernel'
clf = SVC(kernel='linear')

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

print 'Create SVM classfier with rbf kernel for C=10.'
#clf = SVC(kernel='rbf')
clf = SVC(kernel='rbf',C=10.)

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

print 'Create SVM classfier with rbf kernel for C=100.'
#clf = SVC(kernel='rbf')
clf = SVC(kernel='rbf',C=100.)

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

print 'Create SVM classfier with rbf kernel for C=1000.'
#clf = SVC(kernel='rbf')
clf = SVC(kernel='rbf',C=1000.)

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

print "accuracy:", round(clf.score(features_test, labels_test), 3)
print "mislabeled:", (clf.predict(features_test)!=labels_test).sum(), "/", len(labels_test)
print ''
#"""
#########################################################

print 'Create SVM classfier with rbf kernel for C=10000.'
#clf = SVC(kernel='rbf')
clf = SVC(kernel='rbf',C=10000.)

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

print "accuracy:", round(clf.score(features_test, labels_test), 3)
print "mislabeled:", (clf.predict(features_test)!=labels_test).sum(), "/", len(labels_test)
print ''

print '10: ', pred[10]
print '26: ', pred[26]
print '50: ', pred[50]

print 'Chris: ', list(pred).count(1), "/", len(labels_test)
print 'Sara : ', list(pred).count(0), "/", len(labels_test)

#########################################################


