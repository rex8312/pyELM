__author__ = 'rex8312'

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import normalize
from scipy.linalg import pinv2


"""
def normalized(a, axis=-1, order=0):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)
"""

def normalized(X):
    x_max = np.max(X, axis=0)
    x_min = np.min(X, axis=0)
    x_max_ = np.tile(x_max, (X.shape[0], 1))
    x_min_ = np.tile(x_min, (X.shape[0], 1))
    X_ = (X - x_min_) / (x_max_ - x_min_)
    #X_ = 2. * X_ - 1.
    return X_


class BasicExtreamLearningMachine(BaseEstimator, ClassifierMixin):

    def __init__(self, L=100, _lambda=5.0):
        self.L = L
        self.a = None
        self.b = None
        self._lambda = _lambda
        self.g_func = self.sigmoid

    def sigmoid(self, X):
        X1 = 1.0 / (1.0 + np.exp(-X)) #* 2. - 1.
        return X1

    def tanh_func(self, X):
        X1 = (1.0 - np.exp(-2.0 * X)) / (1.0 + np.exp(-2.0 * X))
        return X1

    def _append_bias(self, X):
        return np.append(X, np.ones((X.shape[0], 1)), axis=1)

    def fit(self, X, y):
        X = normalized(X)
        X = self._append_bias(X)
        T = np.array(y)

        self.a = np.random.random((self.L, X.shape[1])) #* 2. - 1.
        #print X.shape, self.a.T.shape
        H = self.g_func(X.dot(self.a.T))
        self.b = pinv2(H).dot(T)

        return self

    def predict(self, X):
        X = normalize(X, axis=0)
        X = self._append_bias(X)
        H = self.g_func(X.dot(self.a.T))
        prediction = H.dot(self.b.T)
        nprediction = normalized(prediction.reshape(prediction.shape[0], 1))
        rs = np.array(np.ones(nprediction.shape) * 0.5 < nprediction, dtype=int)
        return rs
