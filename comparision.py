# -*- coding: utf-8 -*-

__author__ = 'rex8312'

from sklearn import datasets
from sklearn import cross_validation
from time import time
from datetime import timedelta

from plot_comparison import make_classifiers

import numpy as np


if __name__ == '__main__':

    ds = list()
    data = datasets.load_iris()
    ds.append(('iris', data.data, data.target))
    data = datasets.load_digits()
    ds.append(('digits', data.data, data.target))
    X, y = datasets.make_classification(n_samples=1000, n_features=20, n_classes=2)
    ds.append(('gen', X, y))

    classifiers = make_classifiers()

    for data_name, X, y in ds:
        print '*'*10 + ' ' + data_name + ' ' + '*'*10
        for clf_name, clf in classifiers:
            start = time()
            scores = list()
            for _ in range(10):
                scores.extend(cross_validation.cross_val_score(clf, X, y, cv=10))
            scores = np.array(scores)
            end = time()
            print("%30s: %10s: Accuracy: %0.3f (+/- %0.3f): %s" % (
                clf_name, data_name, scores.mean(), scores.std(), timedelta(seconds=end-start))
            )
        print

