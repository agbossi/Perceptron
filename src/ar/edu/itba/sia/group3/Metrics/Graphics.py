import numpy as np
import matplotlib.pyplot as plt


def converge_metric(iterations, errors):
    epochs = np.arange(1, iterations+1)
    errors = np.append(errors, np.zeros(len(epochs) - len(errors)))
    plt.plot(epochs, errors)
    plt.xlabel('iterations')
    plt.ylabel('errors')
    plt.show()


def print_xor_data_set(data_set):
    plt.scatter(np.array(data_set[:2, 0]), np.array(data_set[:2, 1]), marker='x')  # tener en cuenta que final del intervalo es exclusivo
    plt.scatter(np.array(data_set[2:, 0]), np.array(data_set[2:, 1]), marker='o')
    plt.xlabel('x value')
    plt.ylabel('y value')
    plt.show()


def print_and_data_set(data_set):
    plt.scatter(np.array(data_set[:3, 0]), np.array(data_set[:3, 1]), marker='x')  # tener en cuenta que final del intervalo es exclusivo
    plt.scatter(np.array(data_set[3:, 0]), np.array(data_set[3:, 1]), marker='o')
    plt.xlabel('x value')
    plt.ylabel('y value')
    plt.show()