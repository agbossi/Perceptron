import sys

import numpy as np
from sklearn.utils import shuffle
# falta la normalizacion para el punto dos
from ar.edu.itba.sia.group3.Functions.Other_functions import Normalizator


class Perceptron:
    def __init__(self, features_amount, activation_function, problem_type):
        self.weights = np.random.rand(1, features_amount + 1)  # cantidad de entradas (x1,..,xn) mas un lugar para la constante
        self.min_weights = self.weights
        self.error = sys.maxsize
        self.min_error = sys.maxsize
        self.activation_function = activation_function
        self.problem_type = problem_type
        self.delta = np.zeros(features_amount + 1)
        # variables que no voy a usar hasta que no corra pero sino inicializo aca no se como hacer
        self.restart_condition = 0
        self.restart_count = 0
        self.learning_rate = 0
        self.normalizator = None
        self.last_activation_value = 0

    def run(self, training_example, run_mode):
        x = np.array(training_example[:-1])  # training menos label
        x = np.append(x, [1])  # agrego el 1 del bias
        excitement = np.dot(self.weights, x.transpose())
        activation = self.activation_function.get_value(self.normalizator.normalize(excitement[0]))
        activation = self.normalizator.revert_normalization(activation)
        if self.problem_type == "classification":
            error = classification_error(activation, training_example[-1])
        else:
            error = regression_error(activation, training_example[-1])
        delta = self.learning_rate * (training_example[-1] - activation) * x * self.activation_function.get_derivative(self.normalizator.normalize(excitement[0]))
        if run_mode == "training":
            return delta, error
        else:
            return activation, error

    def run_multilayer(self, sigmas):
        x = np.array(sigmas)
       # x = np.append(x, [1])  # agrego el 1 del bias
        self.last_activation_value = np.dot(self.weights[0], x.transpose())
        # activation = self.activation_function.get_value(self.normalizator.normalize(excitement[0])) TODO
        activation = self.activation_function.get_value(self.last_activation_value)
        # activation = self.normalizator.revert_normalization(activation) TODO

        return activation

    def check_restart(self, restart_condition):
        if self.restart_count > restart_condition:
            self.weights = np.random.rand((len(self.weights)))

    def batch_training(self, training_set, learning_rate, restart_condition, iteration_limit, need_to_normalize=False):  # matriz el conjunto)
        if need_to_normalize:
            self.normalizator = Normalizator(np.amax(training_set[:, -1:]))
        else:
            self.normalizator = Normalizator(1)
        self.restart_condition = restart_condition
        self.learning_rate = learning_rate
        iteration_count = 0
        errors_per_epoch = []
        while self.error > 0 and iteration_count < iteration_limit:
            self.check_restart(restart_condition)
            self.error = 0
            self.delta = np.zeros(len(training_set[0]))
            for training_example in training_set:
                delta, error = self.run(training_example, "training")
                self.error += error
                self.delta += delta
            self.weights = np.add(self.weights, self.delta)
            self.min_error, self.min_weights, self.restart_count, errors_per_epoch = error_handling(errors_per_epoch, self.error, iteration_count, self.min_error, self.min_weights, self.weights, self.restart_count)
            iteration_count += 1
        return self.min_weights, errors_per_epoch

    def incremental_training(self, training_set, learning_rate, restart_condition, iteration_limit, need_to_normalize=False):
        if need_to_normalize:
            self.normalizator = Normalizator(np.amax(training_set[:, :-1]))
        else:
            self.normalizator = Normalizator(1)
        self.restart_condition = restart_condition
        self.learning_rate = learning_rate
        iteration_count = 0
        errors_per_epoch = []
        while self.error > 0 and iteration_count < iteration_limit:
            self.check_restart(restart_condition)
            self.error = 0
            training_set = shuffle(training_set)
            for training_example in training_set:
                self.delta, error = self.run(training_example, "training")
                self.error += error
                self.weights = np.add(self.weights, self.delta)
            self.min_error, self.min_weights, self.restart_count, errors_per_epoch = error_handling(errors_per_epoch, self.error, iteration_count, self.min_error, self.min_weights, self.weights, self.restart_count)
            iteration_count += 1
        return self.min_weights, errors_per_epoch

    def test_perceptron(self, testing_set, silent = False):
        halfwaySquareError = 0
        self.error = 0
        for testing_example in testing_set:
            output, error = self.run(testing_example, "testing")
            if not silent:
                print("neuron answer for parameters: ", np.array_str(testing_example), " is ", output, " real answer is ", testing_example[-1])
            self.error += error
            halfwaySquareError += np.square(testing_example[-1] - output)
        print("n=",len(testing_set))
        return self.error, halfwaySquareError/(len(testing_set)) # TODO esto esta ok? o dividido dos? en google es con N asi


def error_handling(errors, error, iteration, min_error, min_weights, weights, cost_count):
    errors.append(error)  # cuantos se clasificaron mal en esta epoca
    if error < min_error:
        min_error = error
        min_weights = weights
    if iteration != 0:
        if errors[iteration] == errors[iteration - 1]:
            cost_count += 1
        else:
            cost_count = 0
    return min_error, min_weights, cost_count, errors


def regression_error(real_output, desired_output):
    if real_output - desired_output:  # lo clasifico mal
        return abs(real_output - desired_output)
    else:
        return 0


def classification_error(real_output, desired_output):
    if real_output - desired_output:  # lo clasifico mal
        return 1
    else:
        return 0


#def batch_perceptron(training_set, iterations, learning_rate, activation_function, problem_type, restart_condition):
#    weights = np.random.rand(1, len(training_set[0]))  # 3 pesos, 2 parametros del or + bias/termino constante
#    iteration = 0
#    error = len(training_set)
#    min_error = len(training_set)
#    min_weights = weights
#    cost_count = 0
#    errors = []
#    while error > 0 and iteration < iterations:  # sigue entrando pese a que iteration > iterations
#        if cost_count > restart_condition:  # numero magico arbitrario, cambiar
#            weights = np.random.rand((1, len(training_set[0])))
#        delta = np.zeros(len(training_set[0]))
#        error = 0
#        for training_example in training_set:
#            x = np.array(training_example[:-1])  # training menos label
#            x = np.append(x, [1])  # agrego el 1 del bias
#            excitement = np.dot(weights, x.transpose())
#            activation = activation_function(excitement)
#            if problem_type == "classification":
#                error += classification_error(activation, training_example[-1])
#            else:
#                error += regression_error(activation, training_example[-1])
#            delta = np.add(delta, learning_rate * (training_example[-1] - activation) * x)
#            # lo clasifico mal
#        weights = np.add(weights, delta)
#        min_error, min_weights, cost_count, errors = error_handling(errors, error, iteration, min_error, min_weights, weights, cost_count)
#        iteration += 1
#    return min_weights, errors

# def incremental_perceptron(training_set, iterations, learning_rate, activation_function, problem_type, restart_condition):
#    weights = np.random.rand(1, len(training_set[0])) #3 pesos, 2 parametros del or + bias/termino constante
#    iteration = 0
#    error = len(training_set)
#    min_error = len(training_set)
#    min_weights = weights
#    cost_count = 0
#    errors = []
#    delta = np.zeros(len(training_set[0]))
#    while error > 0 and iteration < iterations:
#        if cost_count > restart_condition:  # numero magico arbitrario, cambiar
#            weights = np.random.rand((1, len(training_set[0])))
#        training_set = shuffle(training_set)
#        error = 0
#        for training_example in training_set:
#            x = np.array(training_example[:-1]) #training menos label
#            x = np.append(x, [1]) #agrego el 1 del bias
#            excitement = np.dot(weights, x.transpose())
#            activation = activation_function.get_value(excitement[0])
#            if problem_type == "classification":
#                error += classification_error(activation, training_example[-1])
#            else:
#                error += regression_error(activation, training_example[-1])
#            delta = learning_rate * (training_example[-1] - activation) * x * activation_function.get_derivative(excitement[0])
#            weights = np.add(weights, delta)
#        min_error, min_weights, cost_count, errors = error_handling(errors, error, iteration, min_error, min_weights, weights, cost_count)
#        iteration += 1
#    return min_weights, errors
