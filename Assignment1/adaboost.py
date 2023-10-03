import ConfigSpace
import numpy as np
import sklearn.datasets
import sklearn.model_selection
from sklearn.ensemble import AdaBoostClassifier
import typing

from assignment import SequentialModelBasedOptimization

class Adaboost():
    def __init__(self):
        self.param_dict = {"learning_rate": (-3, 0, True)}

    def optimize(self, hp: list, data):

        X_train, X_valid, y_train, y_valid = data
        lr = hp[0]

        clf = AdaBoostClassifier(learning_rate=lr)
        clf.fit(X_train, y_train)
        return sklearn.metrics.accuracy_score(y_valid, clf.predict(X_valid))

    def sample_configurations(self, n: int):
        gamma = 10 ** np.random.uniform(-3, 0, n)
        config = np.empty((n, 1))
        config[:, 0] = gamma
        return config

    def get_algorithm(self):

        return AdaBoostClassifier()


