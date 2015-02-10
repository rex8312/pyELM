__author__ = 'rex8312'

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import normalize


class BasicExtreamLearningMachine(BaseEstimator, ClassifierMixin):

    def __init__(self, L=100, _lambda=1.0):
        self.L = L
        self.a = None
        self.b = None
        self._lambda = _lambda
        self.g_func = self.logistic_func

    def logistic_func(self, X):
        X1 = 1.0 / (1. + np.exp(-X))
        return X1

    def tanh_func(self, X):
        X1 = (1.0 - np.exp(-2.0 * X)) / (1.0 + np.exp(-2.0 * X))
        return X1

    def _append_bias(self, X):
        return np.append(X, np.ones((X.shape[0], 1)), axis=1)

    def fit(self, X, y):
        X = normalize(X, axis=0)
        X = self._append_bias(X)
        T = y

        self.a = np.random.random((self.L, X.shape[1]))
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
        X = normalize(X, axis=0)
        X = self._append_bias(X)
        H = self.g_func(X.dot(self.a.T))
        prediction = self.g_func(H.dot(self.b.T))

        max_p = np.max(prediction)
        min_p = np.min(prediction)
        nprediction = prediction / (max_p - min_p) - min_p
        from copy import deepcopy
        nprediction = normalize(deepcopy(prediction), axis=0)
        # http://scikit-learn.org/stable/modules/preprocessing.html
        #1: (0.83-0.28) = x: 0.80

        print max_p, min_p
        for i, _y in enumerate(zip(prediction, nprediction)):
            print i, _y
        exit()
        return prediction.tolist()
