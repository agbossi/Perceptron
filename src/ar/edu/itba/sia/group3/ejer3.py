import numpy as np
# TODO hacer menos canceroso el codigo

import ar.edu.itba.sia.group3.Metrics.Graphics as mtr
import ar.edu.itba.sia.group3.Models.Perceptron_neuron as md  # no puedo traer el paquete?
import ar.edu.itba.sia.group3.Models.Multi_layer_perceptron as mdL
import ar.edu.itba.sia.group3.Functions.Activation_Functions as af
import src.ar.edu.itba.sia.group3.Resamplers.Train_test_split as rs
import src.ar.edu.itba.sia.group3.Resamplers.K_fold_cross_validation as rk
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

activation_function = af.SigmoidFunction(0.5)
#activation_function = af.StepFunction()

features = 2
iteration_limit = 40
restart_condition = 15
learning_rate = 0.2
## Set up layers for neural network
layer_info_list = [
    mdL.LayerInfo(activation_function, 2, 2),
    mdL.LayerInfo(activation_function, 1, 2)
]

## Create and runs neural network
p = mdL.MultiLayerPerceptron(2, layer_info_list, 1, learning_rate)
p.incremental_training(xor_data_set, iteration_limit)
p.test(xor_data_set)
# mtr.converge_metric(iteration_limit, errors_per_epoch)  # exploto porque errors aparecio con longitud 11
# p.test_perceptron(and_data_set)



