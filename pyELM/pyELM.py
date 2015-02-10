__author__ = 'rex8312'

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Perceptron


class BasicExtreamLearningMachine(BaseEstimator, ClassifierMixin):

    def __init__(self, L=100, _lambda=100.0):
        self.L = L
        self.a = None
        self.b = None
        self._lambda = _lambda

    def g_func(self, X):
        X1 = 1. / (1. + np.exp(-X))
        return X1

    def _append_bias(self, X):
        return np.append(X, np.ones((X.shape[0], 1)), axis=1)

    def fit(self, X, y):
        X = self._append_bias(X)
        T = y

        self.a = np.random.random((self.L, X.shape[1])) * 2.0 - 1.0
        print self.a.shape
        H = self.g_func(X.dot(self.a.T))

        try:
            HH = np.mat(H.dot(H.T))
            iHH = np.mat(1./self._lambda + HH).getI()
            self.b = H.T.dot(iHH).dot(T)
        except np.linalg.linalg.LinAlgError as e:
            HH = np.mat(H.T.dot(H))
            iHH = np.mat(1./self._lambda + HH).getI()
            self.b = iHH.dot(H.T).dot(T)

        return self

    def predict(self, X):
        X = self._append_bias(X)
        H = self.g_func(X.dot(self.a.T))
        y_0 = self.g_func(H.dot(self.b.T))
        y_1 = list()
        for y in y_0:
            if y > 0.5:
                y_1.append(1.0)
            else:
                y_1.append(0.0)
        return y_1
