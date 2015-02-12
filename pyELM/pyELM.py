__author__ = 'rex8312'

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv2


class BasicExtreamLearningMachine(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.L = None
        self.a = None
        self.b = None
        self.g_func = np.tanh

    def _append_bias(self, X):
        return np.append(X, np.ones((X.shape[0], 1)), axis=1)

    def _set_L(self, X):
        #self.L = X.shape[0]
        self.L = int(X.shape[0] / 7.)

    def fit(self, X, y):
        stdsc = StandardScaler()
        X = stdsc.fit_transform(X)
        X = self._append_bias(X)
        self._set_L(X)

        self.classes_ = np.unique(y)
        self.n_class = len(self.classes_)

        self.binarizer = LabelBinarizer(-1, 1)
        T = self.binarizer.fit_transform(y)

        self.a = np.random.random((self.L, X.shape[1])) * 2.0 - 1.0
        H = self.g_func(X.dot(self.a.T))
        self.b = pinv2(H).dot(T)
        return self

    def decision_function(self, X):
        stdsc = StandardScaler()
        X = stdsc.fit_transform(X)
        X = self._append_bias(X)

        H = self.g_func(X.dot(self.a.T))
        raw_prediction = H.dot(self.b)
        normalized_prediction = stdsc.fit_transform(raw_prediction)
        return normalized_prediction

    def predict(self, X):
        raw_prediction = self.decision_function(X)
        class_prediction = self.binarizer.inverse_transform(raw_prediction)
        return class_prediction
