# returns a list of list of k-1 training sets and one testing set. if k = 1 returns classic test - train
from ar.edu.itba.sia.group3.Resamplers.Train_test_split import train_test_split
import numpy as np


def cross_validation_split(k, data_set):
    if k == 1:
        return train_test_split(data_set)
    fold_size = int(len(data_set) / k)
    begin = 0
    end = fold_size
    fold_set = []
    for i in range(k):  # si ds es por ejemplo 10, k es 3, hay un dato que me queda fuera de todos, me lo meto en orto?
        training_set = np.array([[0, 0, 0, 0]])  # para que concatenate no explote, no encontre algo mejor
        for j in range(k):
            if i != j:
                training_set = np.concatenate((training_set, data_set[j*fold_size:(j+1)*fold_size, :]), 0)
        fold = [training_set[1:, :], data_set[i*fold_size:(i+1)*fold_size, :]]
        fold_set.append(fold)
    return fold_set


