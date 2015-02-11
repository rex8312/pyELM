# -*- coding: utf-8 -*-

__author__ = 'rex8312'

from sklearn import datasets
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from time import time
from datetime import timedelta

from pyELM.pyELM import BasicExtreamLearningMachine
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

import numpy as np


if __name__ == '__main__':

    #data = datasets.load_iris()
    data = datasets.load_digits()
    X, y = data.data, data.target
    # X, y = datasets.make_classification(n_samples=1000, n_features=20, n_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    classifiers = [
        BasicExtreamLearningMachine,
        DecisionTreeClassifier,
        Perceptron,
        SVC,
        LinearSVC,
    ]

    for classifier in classifiers:
        scores = list()
        start = time()
        for _ in range(1):
            clf = classifier()
            scores.extend(cross_validation.cross_val_score(clf, X, y, cv=10))
        scores = np.array(scores)
        end = time()
        print("%30s: Accuracy: %0.2f (+/- %0.2f): %s" % (
            clf.__class__.__name__, scores.mean(), scores.std(), timedelta(seconds=end-start))
        )

