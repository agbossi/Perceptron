import numpy as np
import matplotlib.pyplot as plt


or_function = [['x', 'y', 'or'], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0]]
and_function = [['x', 'y', 'or'], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1]]
xor_function = [['x', 'y', 'or'], [1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 1, 1]]


def step_function(value):
    if value > 0:
        return 1
    else:
        return 0


def perceptron(training_set, iterations, learning_rate, activation_function):
    weights = np.random.rand(1, 3) #3 pesos, 2 parametros del or + bias/termino constante
    i = 0
    error = 4
    min_error = len(training_set)
    min_weights = weights
    errors = []
    while error > 0 & i < iterations:
    #   if i > iterations/2: #este criterio es medio choto, podria preguntarse
    #       weights = np.random.rand(1, 3)
        delta = np.zeros(3)    #aca no estoy haciendo lo de tomar ejemplos aleatorios. lo hago por lote
        error = 0
        for training_example in training_set:
            x = np.array(training_example[:-1]) #training menos label
            x = np.append(x, [1]) #agrego el 1 del bias
            excitement = np.dot(weights, x.transpose())
            activation = activation_function(excitement)
            delta = np.add(delta, learning_rate * (training_example[-1] - activation) * x)
            if training_example[-1] - activation: #lo clasifico mal
                error += 1
        weights = np.add(weights, delta)
        errors.append(error) #cuantos se clasificaron mal en esta epoca
        if error < min_error:
            min_error = error
            min_weights = weights
        i += 1
    return min_weights, errors


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


def test_perceptron(weights, testing_set, activation_function):
    for testing_example in testing_set:
        x = np.array(testing_example)  # training menos label
        x = np.append(x, [1])  # agrego el 1 del bias
        excitement = np.dot(weights, x.transpose())
        activation = activation_function(excitement)
        print("answer for parameters: ", np.array_str(testing_example), " is ",  activation)


and_data_set = np.array(and_function[1:])

#print_and_data_set(and_data_set)
#trained_weights, errors_per_epoch = perceptron(and_data_set, 10, 0.1, step_function)
#converge_metric(10, errors_per_epoch) #exploto porque errors aparecio con longitud 11
test_perceptron([[0.27977226,  0.25567711, -0.45243617]], and_data_set[:, :-1], step_function)









