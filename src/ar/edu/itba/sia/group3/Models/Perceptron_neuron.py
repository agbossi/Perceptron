import numpy as np
from sklearn.utils import shuffle


def batch_perceptron(training_set, iterations, learning_rate, activation_function):
    weights = np.random.rand(1, len(training_set[0]))  # 3 pesos, 2 parametros del or + bias/termino constante
    i = 0
    error = len(training_set)
    min_error = len(training_set)
    min_weights = weights
    cost_count = 0
    errors = []
    while error > 0 & i < iterations:
        if cost_count > 5:  # numero magico arbitrario, cambiar
            weights = np.random.rand((1, len(training_set[0])))
        delta = np.zeros(len(training_set[0]))
        error = 0
        for training_example in training_set:
            x = np.array(training_example[:-1])  # training menos label
            x = np.append(x, [1])  # agrego el 1 del bias
            excitement = np.dot(weights, x.transpose())
            activation = activation_function(excitement)
            delta = np.add(delta, learning_rate * (training_example[-1] - activation) * x)
            # lo clasifico mal
            if training_example[-1] - activation:
                error += 1
        weights = np.add(weights, delta)
        errors.append(error)  # cuantos se clasificaron mal en esta epoca
        if error < min_error:
            min_error = error
            min_weights = weights
        if i != 0:
            if errors[i] == errors[i-1]:
                cost_count += 1
            else:
                cost_count = 0
        i += 1
    return min_weights, errors


def incremental_perceptron(training_set, iterations, learning_rate, activation_function):
    weights = np.random.rand(1, len(training_set[0])) #3 pesos, 2 parametros del or + bias/termino constante
    i = 0
    error = len(training_set)
    min_error = len(training_set)
    min_weights = weights
    cost_count = 0
    errors = []
    delta = np.zeros(len(training_set[0]))
    while error > 0 & i < iterations:
        if cost_count > 5:  # numero magico arbitrario, cambiar
            weights = np.random.rand((1, len(training_set[0])))
        training_set = shuffle(training_set)
        error = 0
        for training_example in training_set:
            x = np.array(training_example[:-1]) #training menos label
            x = np.append(x, [1]) #agrego el 1 del bias
            excitement = np.dot(weights, x.transpose())
            activation = activation_function(excitement)
            delta = learning_rate * (training_example[-1] - activation) * x
            weights = np.add(weights, delta)
            if training_example[-1] - activation: #lo clasifico mal
                error += 1
        errors.append(error)  # cuantos se clasificaron mal en esta epoca
        if error < min_error:
            min_error = error
            min_weights = weights
        if i != 0:
            if errors[i] == errors[i - 1]:
                cost_count += 1
            else:
                cost_count = 0
        i += 1
    return min_weights, errors


#duplicado del batch. ver como cambiar
def test_perceptron(weights, testing_set, activation_function):
    error = 0
    for testing_example in testing_set:
        activation = test_example_perceptron(weights, testing_example, activation_function)
        if activation - testing_example[-1]:
            error += 1
    return error


def test_example_perceptron(weights, testing_example, activation_function):
    x = np.array(testing_example)  # training menos label
    x = np.append(x, [1])  # agrego el 1 del bias
    excitement = np.dot(weights, x.transpose())
    activation = activation_function(excitement)
    print("answer for parameters: ", np.array_str(testing_example), " is ", activation)
    return activation
