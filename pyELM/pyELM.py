__author__ = 'rex8312'

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Perceptron


class BasicExtreamLearningMachine(BaseEstimator, ClassifierMixin):

    def __init__(self, n=8):
        self.n = n
        self.a = None
        self.b = None
        self.clf = Perceptron()

    def g_func(self, X):
        X1 = 1. / (1. + np.exp(-X))
        return X1

    def fit(self, X, y):
        self.a = np.random.random((self.n, X.shape[1]))
        H = self.g_func(X.dot(self.a.T))
        inv_H = np.matrix(H).getI()
        self.b = inv_H.dot(y)
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        H = self.g_func(X.dot(self.a.T))
        y_0 = self.g_func(H.dot(self.b.T))
        y_1 = list()
        for y in y_0:
            if y > 0.5:
                y_1.append(1.0)
            else:
                y_1.append(0.0)
        return y_1
