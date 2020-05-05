import numpy as np
# TODO hacer menos canceroso el codigo

import ar.edu.itba.sia.group3.Metrics.Graphics as mtr
import ar.edu.itba.sia.group3.Models.Perceptron_neuron as md  # no puedo traer el paquete?
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


# or_function = [['x', 'y', 'or'], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0]]
and_function = [['x', 'y', 'or'], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1]]
# xor_function = [['x', 'y', 'or'], [1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 1, 1]]
and_data_set = np.array(and_function[1:])
#mtr.print_and_data_set(and_data_set)
activation_function = af.SigmoidFunction(0.5)
features = 3
iteration_limit = 40
restart_condition = 5
learning_rate = 0.2
#trained_weights, errors_per_epoch = md.batch_perceptron(and_data_set, iteration_limit, learning_rate, af.step_function, "classification", restart_condition)
p = md.Perceptron(features, activation_function, "regression")
#trained_weights, errors_per_epoch = p.batch_training(and_data_set, learning_rate, restart_condition, iteration_limit)
#mtr.converge_metric(10, errors_per_epoch)  # exploto porque errors aparecio con longitud 11
#p.test_perceptron(and_data_set)

df = load_data_set()
input_list = rs.train_test_split(df, 0.7)
trained_weights, errors_per_epoch = p.batch_training(input_list[0][0], learning_rate, restart_condition, iteration_limit, True)
mtr.converge_metric(iteration_limit, errors_per_epoch)
p.test_perceptron(input_list[0][1])


