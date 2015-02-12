#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from pyELM.pyELM import BasicExtreamLearningMachine
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


def make_classifiers():
    classifiers = [
        ('Decision Tree', DecisionTreeClassifier()),
        ('Random Forrest', RandomForestClassifier(n_estimators=50)),
        ("SVC", SVC()),
        ('Linear SVC', LinearSVC()),
        ("My ELM", BasicExtreamLearningMachine()),
        ('ELM ensmnble 10', BaggingClassifier(base_estimator=BasicExtreamLearningMachine(),
                                              n_estimators=10, max_samples=1.0, max_features=1.0)),
        ('ELM ensmnble 20', BaggingClassifier(base_estimator=BasicExtreamLearningMachine(),
                                              n_estimators=20, max_samples=1.0, max_features=1.0)),
        ('ELM ensmnble 30', BaggingClassifier(base_estimator=BasicExtreamLearningMachine(),
                                              n_estimators=30, max_samples=1.0, max_features=1.0)),
        ]

    return classifiers


def get_data_bounds(X):
    h = .02  # step size in the mesh

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    return (x_min, x_max, y_min, y_max, xx, yy)


def plot_data(ax, X_train, y_train, X_test, y_test, xx, yy):
    cm = ListedColormap(['#FF0000', '#0000FF'])
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())


def plot_contour(ax, X_train, y_train, X_test, y_test, xx, yy, Z):
    cm = pl.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    ax.set_title(name)
    ax.text(xx.max() - 0.3, yy.min() + 0.3, ('%.2f' % score).lstrip('0'), size=13, horizontalalignment='right')


def make_linearly_separable():
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, random_state=1,
                               n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return (X, y)


def make_datasets():
    return [
        make_moons(n_samples=200, noise=0.3, random_state=0),
        make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1),
        make_linearly_separable()
    ]


if __name__ == '__main__':
    datasets = make_datasets()
    classifiers = make_classifiers()

    i = 1
    figure = pl.figure(figsize=(18, 9))

    # iterate over datasets
    for ds in datasets:
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=0)
        x_min, x_max, y_min, y_max, xx, yy = get_data_bounds(X)

        # plot dataset first
        ax = pl.subplot(len(datasets), len(classifiers) + 1, i)
        plot_data(ax, X_train, y_train, X_test, y_test, xx, yy)

        i += 1

        # iterate over classifiers
        for name, clf in classifiers:
            ax = pl.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)

            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will asign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)

            plot_contour(ax, X_train, y_train, X_test, y_test, xx, yy, Z)

            i += 1

    figure.subplots_adjust(left=.02, right=.98)
    pl.show()
