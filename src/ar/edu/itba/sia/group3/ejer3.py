import enum

import numpy as np
import ar.edu.itba.sia.group3.Models.Multi_layer_perceptron as mdL
import ar.edu.itba.sia.group3.Functions.Activation_Functions as af
import pandas as pd
from os.path import expanduser as ospath
from sklearn.utils import shuffle


# ospath('~/PycharmProjects/Perceptron/TP3-ej2-Conjunto_entrenamiento.xlsx')
def load_data_set():
    data = pd.read_excel('TP3-ej2-Conjunto_entrenamiento.xlsx').iloc[1:, :4]  # TODO manejar path de archivo
    data = shuffle(data)
    return data.to_numpy()


xor_function = [['x', 'y', 'or'], [1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 1, 1]]
xor_data_set = np.array(xor_function[1:])

activation_function = af.SigmoidFunction(2)


class ClassificationTypes(enum.Enum):
    cero = 0
    uno = 1


classifications = [ClassificationTypes.cero, ClassificationTypes.uno]

features = 2
iteration_limit = 5000
# restart_condition = 15
learning_rate = 0.5
momentum = 0.5  # cero to disable momentum
## Set up layers for neural network
layer_info_list = [
    mdL.LayerInfo(activation_function, 3, 2),
    mdL.LayerInfo(activation_function, 1, 3)
]

## Create and runs neural network
p = mdL.MultiLayerPerceptron(2, layer_info_list, 1, learning_rate, momentum)
p.incremental_training(xor_data_set, iteration_limit)
p.test_classification(xor_data_set, classifications)
# mtr.converge_metric(iteration_limit, errors_per_epoch)  # exploto porque errors aparecio con longitud 11
# p.test_perceptron(and_data_set)
