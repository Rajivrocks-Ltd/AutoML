import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.model_selection
from sklearn.neural_network import MLPClassifier

from assignment import SequentialModelBasedOptimization

class MLPclassifier():
    def __init__(self):

        self.param_dict = {
            "learning_rate_init": (-3, 0, True, float),
            "beta_1": (0, 0.99, False, float),
            "beta_2": (0, 0.999, False, float),
            "hidden_layer_sizes": (5, 300, False, int)
        }

    def optimize(self, hp: list, data):

        X_train, X_valid, y_train, y_valid = data
        lr, b1, b2, n_hidden = hp[0], hp[1], hp[2], int(hp[3])

        clf = MLPClassifier(learning_rate='constant', learning_rate_init=lr, beta_1=b1, beta_2=b2, hidden_layer_sizes=n_hidden)
        clf.fit(X_train, y_train)
        return sklearn.metrics.accuracy_score(y_valid, clf.predict(X_valid))

    def sample_configurations(self, n: int):
        gamma = 10 ** np.random.uniform(-3, 0, n)
        beta1 = np.random.uniform(0, 1, n)
        beta2 = np.random.uniform(0, 1, n)
        n_hidden = np.random.randint(5, 300, n)

        config = np.empty((n, 4))
        config[:, 0] = gamma
        config[:, 1] = beta1
        config[:, 2] = beta2
        config[:, 3] = n_hidden
        return config

    def get_algorithm(self):
        return MLPClassifier()

