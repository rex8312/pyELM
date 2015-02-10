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
            return -1.0
    return _func

if __name__ == '__main__':
    data = load_iris()
    X = data.data
    y0 = map(func(0), data.target)
    y1 = map(func(1), data.target)
    y2 = map(func(2), data.target)

    L = int(len(X) * 0.9)
    L = 150

    clf = BasicExtreamLearningMachine(L=L)
    clf.fit(X, y0)
    y = clf.predict(X)
    for i, _y in enumerate(y):
        print i, _y
    exit()


    for y in [y0, y1, y2]:
        scores = cross_validation.cross_val_score(clf, X, y, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

