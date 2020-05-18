import numpy as np
# Este esta cerrado !!!

import ar.edu.itba.sia.group3.Metrics.Graphics as mtr
import ar.edu.itba.sia.group3.Models.Perceptron_neuron as md  # no puedo traer el paquete?
import ar.edu.itba.sia.group3.Functions.Activation_Functions as af

and_function = [['x', 'y', 'or'], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1]]
xor_function = [['x', 'y', 'or'], [1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 1, 1]]
and_data_set = np.array(and_function[1:])
xor_data_set = np.array(xor_function[1:])

activation_function = af.StepFunction()
features = 2
iteration_limit = 10
restart_condition = 5
learning_rate = 0.2

p = md.Perceptron(features, activation_function, "classification")
trained_weights, errors_per_epoch = p.batch_training(xor_data_set, learning_rate, restart_condition, iteration_limit)
mtr.converge_metric(iteration_limit, errors_per_epoch)  # exploto porque errors aparecio con longitud 11
p.test_perceptron(xor_data_set)

