import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.model_selection
from sklearn.neural_network import MLPClassifier

from assignment import SequentialModelBasedOptimization

class MLPclassifier():
    def __init__(self):
        pass
    def optimize(self, hp: list, data):

        X_train, X_valid, y_train, y_valid = data
        lr, b1, b2 = hp[0], hp[1], hp[2]

        clf = MLPClassifier(learning_rate='constant', learning_rate_init=lr, beta_1=b1, beta_2=b2)
        clf.fit(X_train, y_train)
        return sklearn.metrics.accuracy_score(y_valid, clf.predict(X_valid))

    def sample_configurations(self, n: int):
        gamma = 10 ** np.random.uniform(-3, 0, n)
        beta1 = np.random.uniform(0, 1, n)
        beta2 = np.random.uniform(0, 1, n)
        config = np.empty((n, 3))
        config[:, 0] = gamma
        config[:, 1] = beta1
        config[:, 2] = beta2
        return config


#data = sklearn.datasets.fetch_openml(name='mnist_784', version=1)
