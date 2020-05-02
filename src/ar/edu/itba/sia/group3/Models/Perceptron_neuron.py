import numpy as np
from sklearn.utils import shuffle


def batch_perceptron(training_set, iterations, learning_rate, activation_function, problem_type, restart_condition):
    weights = np.random.rand(1, len(training_set[0]))  # 3 pesos, 2 parametros del or + bias/termino constante
    iteration = 0
    error = len(training_set)
    min_error = len(training_set)
    min_weights = weights
    cost_count = 0
    errors = []
    while error > 0 & iteration < iterations:
        if cost_count > restart_condition:  # numero magico arbitrario, cambiar
            weights = np.random.rand((1, len(training_set[0])))
        delta = np.zeros(len(training_set[0]))
        error = 0
        for training_example in training_set:
            x = np.array(training_example[:-1])  # training menos label
            x = np.append(x, [1])  # agrego el 1 del bias
            excitement = np.dot(weights, x.transpose())
            activation = activation_function(excitement)
            if problem_type == "classification":
                error = classification_error(activation, training_example[-1])
            else:
                error = regression_error(activation, training_example[-1])
            delta = np.add(delta, learning_rate * (training_example[-1] - activation) * x)
            # lo clasifico mal
        weights = np.add(weights, delta)
        min_error, min_weights, cost_count, errors = error_handling(errors, error, iteration, min_error, min_weights, weights, cost_count)
        iteration += 1
    return min_weights, errors


def incremental_perceptron(training_set, iterations, learning_rate, activation_function, problem_type, restart_condition):
    weights = np.random.rand(1, len(training_set[0])) #3 pesos, 2 parametros del or + bias/termino constante
    iteration = 0
    error = len(training_set)
    min_error = len(training_set)
    min_weights = weights
    cost_count = 0
    errors = []
    delta = np.zeros(len(training_set[0]))
    while error > 0 & iteration < iterations:
        if cost_count > restart_condition:  # numero magico arbitrario, cambiar
            weights = np.random.rand((1, len(training_set[0])))
        training_set = shuffle(training_set)
        error = 0
        for training_example in training_set:
            x = np.array(training_example[:-1]) #training menos label
            x = np.append(x, [1]) #agrego el 1 del bias
            excitement = np.dot(weights, x.transpose())
            activation = activation_function(excitement)
            if problem_type == "classification":
                error += classification_error(activation, training_example[-1])
            else:
                error += regression_error(activation, training_example[-1])
            delta = learning_rate * (training_example[-1] - activation) * x
            weights = np.add(weights, delta)
        min_error, min_weights, cost_count, errors = error_handling(errors, error, iteration, min_error, min_weights, weights, cost_count)
        iteration += 1
    return min_weights, errors


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
        return real_output - desired_output
    else:
        return 0


def classification_error(real_output, desired_output):
    if real_output - desired_output:  # lo clasifico mal
        return 1
    else:
        return 0


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
