# -*- coding: utf-8 -*-

__author__ = 'rex8312'

from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
from sklearn import cross_validation

from pyELM.pyELM import BasicExtreamLearningMachine


def func(idx):
    def _func(x):
        if x == idx:
            return 1.0
        else:
            return 0.0
    return _func

if __name__ == '__main__':
    data = load_iris()
    X = normalize(data.data, axis=0)
    y0 = map(func(0), data.target)
    y1 = map(func(1), data.target)
    y2 = map(func(2), data.target)


    clf = BasicExtreamLearningMachine(n=256)
    for y in [y0, y1, y2]:
        scores = cross_validation.cross_val_score(clf, X, y, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

