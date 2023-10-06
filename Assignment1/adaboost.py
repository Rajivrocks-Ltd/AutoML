import ConfigSpace
import numpy as np
import sklearn.datasets
import sklearn.model_selection
from sklearn.ensemble import AdaBoostClassifier
import typing

from assignment import SequentialModelBasedOptimization

class Adaboost():
    def __init__(self):
        self.param_dict = {
            "learning_rate": (-3, 0, True, float),
            "n_estimators": (1, 300, False, int)
        }

    def optimize(self, hp: list, data):

        X_train, X_valid, y_train, y_valid = data
        lr, n_estimators = hp[0], int(hp[1])

        clf = AdaBoostClassifier(learning_rate=lr, n_estimators=n_estimators)
        clf.fit(X_train, y_train)
        return sklearn.metrics.accuracy_score(y_valid, clf.predict(X_valid))

    def sample_configurations(self, n: int):
        gamma = 10 ** np.random.uniform(-3, 0, n)
        n_estimators = np.random.randint(1, 300, n)

        config = np.empty((n, 2))
        config[:, 0] = gamma
        config[:, 1] = n_estimators

        return config

    def get_algorithm(self):

        return AdaBoostClassifier()


