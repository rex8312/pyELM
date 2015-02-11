__author__ = 'rex8312'

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from scipy.linalg import pinv2


def normalized(X):
    x_max = np.max(X, axis=0)
    x_min = np.min(X, axis=0)
    x_max_ = np.tile(x_max, (X.shape[0], 1))
    x_min_ = np.tile(x_min, (X.shape[0], 1))
    X_ = (X - x_min_) / (x_max_ - x_min_)
    #X_ = 2. * X_ - 1.
    return X_


class BasicExtreamLearningMachine(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.L = None
        self.a = None
        self.b = None
        self.g_func = np.tanh

    #def sigmoid(self, X):
        #X1 = 1.0 / (1.0 + np.exp(-X)) * 2. - 1.
        #return X1

    def _append_bias(self, X):
        return np.append(X, np.ones((X.shape[0], 1)), axis=1)

    def fit(self, X, y, L=None):
        X = self._append_bias(X)
        if L is None:
            self.L = X.shape[0] * 5
        else:
            self.L = L

        enc = OneHotEncoder(categorical_features='all', n_values='auto')
        self.n_class = len(np.unique(y))
        y = np.array(y).reshape((len(y), 1))
        T = enc.fit_transform(y).toarray()

        self.a = np.random.random((self.L, X.shape[1])) * 1.0 - 0.5
        H = self.g_func(X.dot(self.a.T))
        self.b = pinv2(H).dot(T)
        return self

    def predict(self, X):
        X = self._append_bias(X)
        H = self.g_func(X.dot(self.a.T))
        prediction = H.dot(self.b)
        prediction = prediction.reshape(prediction.shape[0], self.n_class)
        #prediction = normalized(prediction)
        #prediction = self.g_func(prediction)
        rs = np.nanargmax(prediction, axis=1)
        return rs
