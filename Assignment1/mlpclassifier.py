import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.model_selection
from sklearn.neural_network import MLPClassifier

from assignment import SequentialModelBasedOptimization

class MLPclassifier():
    def __init__(self):
        pass
    def optimize(self, hp: list):
        clf = MLPClassifier(learning_rate='constant', learning_rate_init=hp[0], beta_1=hp[1], beta_2=hp[2])
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


data = sklearn.datasets.fetch_openml(name='mnist_784', version=1)
X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
    data.data, data.target, test_size=0.33, random_state=1)

X_train = X_train[:5000]
y_train = y_train[:5000]
X_valid = X_valid[:5000]
y_valid = y_valid[:5000]

print(X_train.shape)


ada = MLPclassifier()

configs = ada.sample_configurations(10)
sample_configs = [(config, ada.optimize(config)) for config in configs]
smbo = SequentialModelBasedOptimization()
smbo.initialize(sample_configs)

accuracy = []
for idx in range(16):
    print('iteration %d/16' % idx)
    smbo.fit_model()
    theta_new = smbo.select_configuration(ada.sample_configurations(1024))
    performance = ada.optimize(theta_new)
    smbo.update_runs((theta_new, performance))
    accuracy.append(smbo.theta_inc_performance)
    print(smbo.theta_inc_performance)

plt.plot(accuracy)
plt.show()