# -*- coding: utf-8 -*-

__author__ = 'Hyunsoo Park'

from sklearn.datasets import load_iris, load_digits
from sklearn.datasets import make_moons, make_circles
from plot_comparison import make_linearly_separable
from sklearn import cross_validation
from time import time
from datetime import timedelta

from plot_comparison import make_classifiers

import numpy as np


def make_datasets():
    ds = list()

    data = load_iris()
    ds.append(('iris', data.data, data.target))

    data = load_digits()
    ds.append(('digits', data.data, data.target))

    X, y = make_moons(n_samples=200, noise=0.3, random_state=0)
    ds.append(('moons', X, y))

    X, y = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1)
    ds.append(('circles', X, y))

    X, y = make_linearly_separable()
    ds.append(('linearly separable', X, y))
    return ds

if __name__ == '__main__':

    ds = make_datasets()
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

