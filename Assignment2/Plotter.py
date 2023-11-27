
#-----Imports-----#
import numpy as np
import matplotlib.pyplot as plt
import os


def decode_parameters(save_name):

    param_dict = {}

    save_name = save_name.split("\\")[-1]
    for param_string in save_name.split("-"):

        param_name = ""
        param_value = ""

        value_start = False
        for i in range(len(param_string)):
            char = param_string[i]

            if char.isnumeric() or char == 'F' or (i+1 < len(param_string) and char == 'T' and param_string[i+1] =='r'):
                value_start = True
            if value_start:
                param_value += char
            else:
                param_name += char

        param_dict[param_name] = param_value

    return param_dict

def decode_results(save_name):

    results_dict = {}

    result_names = [x for x in os.listdir(save_name)]

    for result_name in result_names:
        result_path = save_name + "\\" + result_name
        results_dict[result_name[:-4]] = np.load(result_path)

    return results_dict

def read_results(data_folder_name):

    folder_names = [x[0] for x in os.walk(os.getcwd() + "\\" + data_folder_name)][1:]
    param_dicts = [decode_parameters(folder_name) for folder_name in folder_names]
    result_dicts = [decode_results(folder_name) for folder_name in folder_names]

    return param_dicts, result_dicts


def plot_num_ways():

    n_files = len(param_dicts)

    num_ways_list_so = []
    num_ways_list_no_so = []
    accuracy_list_so = []
    accuracy_list_no_so = []

    for i in range(n_files):

        num_ways = int(param_dicts[i]['nw'])
        second_order = param_dicts[i]['so']
        accuracy = float(result_dicts[i]['test-accuracy'][-1])

        if second_order == 'True':
            num_ways_list_so.append(num_ways)
            accuracy_list_so.append(accuracy)
        else:
            num_ways_list_no_so.append(num_ways)
            accuracy_list_no_so.append(accuracy)

    plt.figure()

    plt.scatter(num_ways_list_no_so, accuracy_list_no_so, label="First order")
    plt.scatter(num_ways_list_so, accuracy_list_so, label="Second order")


    plt.xlim(0, 15)
    plt.ylim(0.3, 1.05)

    plt.xlabel("Classes per task")
    plt.ylabel("Test accuracy")

    plt.grid(linestyle="--", alpha=0.5)
    plt.legend(fontsize=8)

    plt.show()



param_dicts, result_dicts = read_results("Task3Results")
plot_num_ways()