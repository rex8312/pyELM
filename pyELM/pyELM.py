# -*- coding: utf-8 -*-

__author__ = 'Hyunsoo Park'

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv2


class BasicExtreamLearningMachine(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.L = None  # 은닉노드 개수
        self.a = None  # 입력-은닉 노드 사이의 가중치
        self.b = None  # 은닉-출력 노드 사이의 가중치
        self.g_func = np.tanh  # 은닉노드의 활성화 함수 tanh을 사용함

    def _append_bias(self, X):
        # 입력의 마지막에 1.을 추가
        return np.append(X, np.ones((X.shape[0], 1)), axis=1)

    def _set_L(self, X):
        #self.L = X.shape[0]
        # 은닉노드 개수 설정
        # 데이터 개수의 1/7 개의 은닉노드를 사용한다.
        # 딱히 이렇게 정한 근거는 없다.
        self.L = int(X.shape[0] / 7.)

    def fit(self, X, y):
        # 입력 데이터 준비
        # [-1, 1] 사이로 정규화하고, 바이어스를 추가함
        stdsc = StandardScaler()
        X = stdsc.fit_transform(X)
        X = self._append_bias(X)
        
        # 은닉노드 개수 설정
        self._set_L(X)

        # target 데이터(클래스) 준비
        self.classes_ = np.unique(y)
        self.n_class = len(self.classes_)
        self.binarizer = LabelBinarizer(-1, 1)
        T = self.binarizer.fit_transform(y)

        # 학습단계
        # 1. 무작위로 입력-은닉노드 사이 가중치 설정
        self.a = np.random.random((self.L, X.shape[1])) * 2.0 - 1.0
        # 2. 은닉노드의 출력 H 구함
        H = self.g_func(X.dot(self.a.T))
        # 3. 은닉-출력 계층 사이 가중치 b 구함: penrose moore 역행렬 (pinv2) 이용
        self.b = pinv2(H).dot(T)
        return self

    def decision_function(self, X):
        stdsc = StandardScaler()
        X = stdsc.fit_transform(X)
        X = self._append_bias(X)

        H = self.g_func(X.dot(self.a.T))
        raw_prediction = H.dot(self.b)
        normalized_prediction = stdsc.fit_transform(raw_prediction)
        class_prediction = self.binarizer.inverse_transform(normalized_prediction)
        return class_prediction

    def predict(self, X):
        return self.decision_function(X)
