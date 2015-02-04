# -*- coding: utf-8 -*-

__author__ = 'rex8312'

from pyELM.pyELM import ExtreamLearningMachine


if __name__ == '__main__':
    clf = ExtreamLearningMachine()

    X, y = ['foo', 'bar', 'foo'], [1, 0, 1]
    clf.fit(X, y)
    print clf.predict(X)


