import matplotlib.pyplot as plt
import numpy as np
import os


def load_numpy(path: str):
    """
        Loads a numpy array from the given path.
    """
    return np.load(path)


def plot_val_loss(data, data2, title):
    """ Plots the validation loss for each validation interval. for two different models. """
    # Calculate the average accuracy for each validation interval
    mean_loss = data.mean(axis=1)
    mean_loss2 = data2.mean(axis=1)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(mean_loss, marker='o', color='blue', linestyle='-', label='Mean val loss 1st order')
    plt.plot(mean_loss2, marker='o', color='orange', linestyle='-', label='Mean val loss 2nd order')
    plt.title(title)
    plt.xlabel('Validation Interval')
    plt.ylabel('Average loss')
    plt.grid(True)
    plt.show()


def plot_train_loss(data, data2, title, window_size=100):
    # Calculate moving averages
    def moving_average(series, size):
        return np.convolve(series, np.ones(size) / size, mode='valid')

    ma_data = moving_average(data, window_size)
    ma_data2 = moving_average(data2, window_size)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(ma_data, color='blue', linestyle='-', label='MA Train Loss 1st Order')
    plt.plot(ma_data2, color='orange', linestyle='-', label='MA Train Loss 2nd Order')

    # Plot original data with transparency
    # plt.plot(data, marker='o', color='blue', linestyle='none', alpha=0.01)
    # plt.plot(data2, marker='o', color='orange', linestyle='none', alpha=0.01)

    plt.title(title)
    plt.xlabel('Training Episodes (after moving average)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


workdir = os.getcwd()

t1first_order = workdir + "\\results\\nw5-nss1-nqs15-mbs1-vi500-net1000-nte40000-l0.001-il0.4-soFalse-domniglot-T1-is28-rFalse-d0-s0 - first order T1"
t1second_order = workdir + "\\results\\nw5-nss1-nqs15-mbs1-vi500-net1000-nte40000-l0.001-il0.4-soTrue-domniglot-T1-is28-rFalse-d0-s0 - second order T1"

t3first_order = workdir + "\\results\\nw5-nss1-nqs15-mbs1-vi500-net1000-nte40000-l0.001-il0.4-soFalse-domniglot-T3-is28-rFalse-d0-s0 - first order T3"
t3second_order = workdir + "\\results\\nw5-nss1-nqs15-mbs1-vi500-net1000-nte40000-l0.001-il0.4-soTrue-domniglot-T3-is28-rFalse-d0-s0 - second order T3"

t6first_order = workdir + "\\results\\nw5-nss1-nqs15-mbs1-vi500-net1000-nte40000-l0.001-il0.4-soFalse-domniglot-T6-is28-rFalse-d0-s0 - first order T6"
t6second_order = workdir + "\\results\\nw5-nss1-nqs15-mbs1-vi500-net1000-nte40000-l0.001-il0.4-soTrue-domniglot-T6-is28-rFalse-d0-s0 - second order T6"

t1_fo_val_loss = f'{t1first_order}\\val-loss.npy'
t1_fo_train_loss = f'{t1first_order}\\train-loss.npy'
t1_so_val_loss = f'{t1second_order}\\val-loss.npy'
t1_so_train_loss = f'{t1second_order}\\train-loss.npy'

t1_fo_val_loss = load_numpy(t1_fo_val_loss)
t1_so_val_loss = load_numpy(t1_so_val_loss)

t1_fo_train_loss = load_numpy(t1_fo_train_loss)
t1_so_train_loss = load_numpy(t1_so_train_loss)

plot_val_loss(t1_fo_val_loss, t1_so_val_loss, "first/second order validation loss")
plot_train_loss(t1_fo_train_loss, t1_so_train_loss, "first/second order training loss")
# print(t1_fo_train_loss.shape)
