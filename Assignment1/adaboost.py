import ConfigSpace
import numpy as np
import sklearn.datasets
import sklearn.model_selection
from sklearn.ensemble import AdaBoostClassifier
import typing

from assignment import SequentialModelBasedOptimization

class Adaboost():
    def __init__(self):
        pass

    def optimize(self, hp: list):
        clf = AdaBoostClassifier(learning_rate=hp[0])
        clf.fit(X_train, y_train)
        return sklearn.metrics.accuracy_score(y_valid, clf.predict(X_valid))

    def sample_configurations(self, n: int):
        gamma = 10 ** np.random.uniform(-3, 0, n)
        config = np.empty((n, 1))
        config[:, 0] = gamma
        return config


data = sklearn.datasets.fetch_openml(name='mnist_784', version=1)
X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
    data.data, data.target, test_size=0.33, random_state=1)

X_train = X_train[:5000]
y_train = y_train[:5000]
X_valid = X_valid[:5000]
y_valid = y_valid[:5000]

print(X_train.shape)


ada = Adaboost()

configs = ada.sample_configurations(10)
sample_configs = [(config, ada.optimize(config)) for config in configs]
smbo = SequentialModelBasedOptimization()
smbo.initialize(sample_configs)

for idx in range(16):
    print('iteration %d/16' % idx)
    smbo.fit_model()
    theta_new = smbo.select_configuration(ada.sample_configurations(1024))
    performance = ada.optimize(theta_new)
    smbo.update_runs((theta_new, performance))
    print(smbo.theta_inc_performance)