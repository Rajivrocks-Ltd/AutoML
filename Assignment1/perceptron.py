import ConfigSpace
import numpy as np
import sklearn.datasets
import sklearn.model_selection
import sklearn.svm
import typing

from assignment import SequentialModelBasedOptimization

class Perceptron():
    def __init__(self):
        pass

    def optimize(self, hp: list, data):

        X_train, X_valid, y_train, y_valid = data
        eta0 = hp[0]

        clf = sklearn.linear_model.Perceptron(eta0 = eta0, max_iter=100)
        clf.fit(X_train, y_train)
        return sklearn.metrics.accuracy_score(y_valid, clf.predict(X_valid))

    def sample_configurations(self, n: int):
        gamma = 10 ** np.random.uniform(-3, 0, n)
        config = np.empty((n, 1))
        config[:, 0] = gamma
        return config

