
#-----Imports-----#
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from assignment import SequentialModelBasedOptimization
from adaboost import Adaboost
from mlpclassifier import MLPclassifier
from perceptron import Perceptron

#Class that performs hyperparameter search for given model on given dataset
class AutoML:

    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    #Perform hyperparameter search
    def run(self, n_iter: int, n_sample: int, n_sample_start: int, data_fraction: float = 1):

        dataset, model = self.dataset, self.model

        #Split the dataset in training and validation set
        data = sklearn.model_selection.train_test_split(
            dataset.data, dataset.target, test_size=0.33, random_state=1)

        if(data_fraction != 1):

            X_train, X_valid, y_train, y_valid = data

            train_length = int(data_fraction * len(X_train))
            valid_length = int(data_fraction * len(X_valid))
            X_train, X_valid, y_train, y_valid = (X_train[:train_length], X_valid[:valid_length],
                                                  y_train[:train_length], y_valid[:valid_length])

            data = [X_train, X_valid, y_train, y_valid]

        #Initial hyperparameter vectors used to initialize the gaussian model
        configs = model.sample_configurations(n_sample_start)
        sample_configs = [(config, model.optimize(config, data)) for config in configs]

        smbo = SequentialModelBasedOptimization()
        smbo.initialize(sample_configs)

        self.accuracy_list = [smbo.theta_inc_performance]

        for idx in range(n_iter):
            print(f"Iteration {idx} of {n_iter}")

            #Fit the gaussian model to predict hyperparameter performance
            smbo.fit_model()

            #Select a new hyperparameter vector to be tested
            theta_new = smbo.select_configuration(model.sample_configurations(n_sample))

            #Now find the performance of these hyperparameters
            performance = model.optimize(theta_new, data)
            smbo.update_runs((theta_new, performance))

            self.accuracy_list.append(smbo.theta_inc_performance)

        self.smbo = smbo

    def plot(self):

        plt.figure()

        plt.plot(range(len(self.accuracy_list)), self.accuracy_list)

        plt.xlim(0, len(self.accuracy_list) - 1)

        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")

        plt.grid(linestyle="--", alpha=0.5)

        plt.show()



if __name__ == "__main__":

    dataset = sklearn.datasets.fetch_openml(name='diabetes', version=1)

    model = Adaboost()

    automl = AutoML(dataset, model)

    automl.run(2000, 1000, 5, 0.5)
    automl.plot()






