# -*- coding: utf-8 -*-

__author__ = 'rex8312'

from sklearn.datasets import load_iris
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

from pyELM.pyELM import BasicExtreamLearningMachine


def func(idx):
    def _func(x):
        if x == idx:
            return 1
        else:
            return 0
    return _func

if __name__ == '__main__':
    stdsc = StandardScaler()
    data = load_iris()
    X = stdsc.fit_transform(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    """
    clf = BasicExtreamLearningMachine()
    clf.fit(X_train, y_train)
    yo = clf.predict(X_test)
    correct = 0.
    incorrect = 0.
    for i, _y in enumerate(zip(y_test, yo)):
        print i, _y
        if _y[0] == _y[1]:
            correct += 1
        else:
            incorrect += 1

    print correct / (correct + incorrect)
#    exit()
    """

    clf = BasicExtreamLearningMachine()
    scores = cross_validation.cross_val_score(clf, X, y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

