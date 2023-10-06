# -----Imports-----#
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

from assignment import SequentialModelBasedOptimization
from adaboost import Adaboost
from mlpclassifier import MLPclassifier
from perceptron import Perceptron


# Class that performs hyperparameter search for given model on given dataset
class AutoML:

    def __init__(self, dataset, model, ):
        self.dataset = dataset
        self.model = model
        self.data = sklearn.model_selection.train_test_split(
            dataset.data, dataset.target, test_size=0.33, random_state=1)

    def reduce_data(self, data_fraction):

        # Split the dataset in training and validation set
        if (data_fraction != 1):
            X_train, x_valid, y_train, y_valid = self.data

            train_length = int(data_fraction * len(X_train))
            valid_length = int(data_fraction * len(x_valid))
            X_train, x_valid, y_train, y_valid = (X_train[:train_length], x_valid[:valid_length],
                                                  y_train[:train_length], y_valid[:valid_length])

            self.data = [X_train, x_valid, y_train, y_valid]


    # Perform hyperparameter search
    def run_bo(self, n_iter: int, n_sample: int, n_sample_start: int, data_fraction: float = 1):

        model = self.model

        # Initial hyperparameter vectors used to initialize the gaussian model
        configs = model.sample_configurations(n_sample_start)
        sample_configs = [(config, model.optimize(config, self.data)) for config in configs]

        smbo = SequentialModelBasedOptimization()
        smbo.initialize(sample_configs)

        self.accuracy_list = [smbo.theta_inc_performance]

        for idx in range(n_iter):
            print(f"Iteration {idx} of {n_iter}")

            # Fit the gaussian model to predict hyperparameter performance
            smbo.fit_model()

            # Select a new hyperparameter vector to be tested
            theta_new = smbo.select_configuration(model.sample_configurations(n_sample))

            # Now find the performance of these hyperparameters
            performance = model.optimize(theta_new, self.data)
            smbo.update_runs((theta_new, performance))

            self.accuracy_list.append(smbo.theta_inc_performance)

        self.smbo = smbo

    def run_gs(self, n_hp: np.array):

        param_grid = {}
        for i, key in enumerate(model.param_dict):

            value = model.param_dict[key]

            use_log = value[2]
            hp_range = np.linspace(value[0], value[1], n_hp[i], dtype= value[3])
            if use_log:
                param_grid[key] = 10 ** hp_range
            else:
                param_grid[key] = hp_range

        clf = sklearn.model_selection.GridSearchCV(self.model.get_algorithm(), param_grid, verbose=10)

        X_train, X_valid, y_train, y_valid = self.data

        clf.fit(X_train, y_train)

        print(sklearn.metrics.accuracy_score(y_valid, clf.predict(X_valid)))

    def run_rs(self, n_iter):

        theta_best = model.sample_configurations(1)[0]
        performance_best = model.optimize(theta_best, self.data)

        for i in range(n_iter - 1):

            print(f"Iteration {i} of {n_iter}")

            theta_new = model.sample_configurations(1)[0]
            performance = model.optimize(theta_new, self.data)

            if(performance > performance_best):
                theta_best = theta_new
                performance_best = performance

        return performance_best


    def plot(self):

        plt.figure()

        plt.plot(range(len(self.accuracy_list)), self.accuracy_list)

        plt.xlim(0, len(self.accuracy_list) - 1)

        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")

        plt.grid(linestyle="--", alpha=0.5)

        plt.show()


if __name__ == "__main__":

    #Adaboost config
    #RS: 225, GS: [15,15], BO: 200, 1000, 25

    #MLP config
    #RS: 256, GS: [4,4,4,4], BO: 231, 1000, 25

    np.random.seed(5)

    dataset = sklearn.datasets.fetch_openml(name='wdbc', version=1)
    model = Adaboost()

    automl = AutoML(dataset, model)
    automl.reduce_data(1)

    print(automl.run_rs(225))

    start = perf_counter()
    automl.run_gs([15, 15])
    end = perf_counter()

    # print(f"gridsearch: {end - start}")

    start = perf_counter()
    automl.run_bo(200, 1000, 25)
    end = perf_counter()
    # print(f"baysian: {end - start}")
    #

    automl.plot()
