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
#and_function = [['x', 'y', 'or'], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1]]
# xor_function = [['x', 'y', 'or'], [1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 1, 1]]
#and_data_set = np.array(and_function[1:])
#mtr.print_and_data_set(and_data_set)
#trained_weights, errors_per_epoch = md.incremental_perceptron(and_data_set, 10, 0.15, af.step_function, "classification", 5)
#mtr.converge_metric(10, errors_per_epoch)  # exploto porque errors aparecio con longitud 11
#md.test_perceptron(trained_weights, and_data_set[:, :-1], af.step_function)

df = load_data_set()
input_list = rs.train_test_split(df, 0.7)
trained_weights, errors_per_epoch = md.batch_perceptron(input_list[0][0], 15, 0.15, af.sigmoid_function, "regression", 5)
mtr.converge_metric(10, errors_per_epoch)
md.test_perceptron(trained_weights, input_list[0][1], af.sigmoid_function())


