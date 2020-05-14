import numpy as np

from ar.edu.itba.sia.group3.Models.Perceptron_neuron import Perceptron


class MultiLayerPerceptron:
    def __init__(self, input_neurons, neurons_info, output_neurons):  # hidden_neurons_info pensado como lista de layerInfo. cada lista con cantidad de neuronas e info para neuronas de cada capa
        self.features = input_neurons
        self.hidden_layers = len(neurons_info)
        self.layers = []
        for layer_info in neurons_info:
            layer = Layer(layer_info.neurons_amount, layer_info.connections, layer_info.activation_function)
            self.layers.append(layer)
        self.targets = output_neurons

    def feed_forward(self, training_example):
        ## aca voy hacia adelante
        elem = [training_example[:-1].tolist()]
        for i in range(len(self.layers)):
            elem.append(np.zeros(len(self.layers[i].neurons)).tolist())
            for j in range(len(self.layers[i].neurons)):
                elem[i + 1][j] = self.layers[i].neurons[j].run_multilayer(np.array(elem[i]))
        return elem

    def back_propagation(self, training_example, elem):
        ## ya tengo en elem la salida de la ultima capa, ahora corrijo pesos hacia atras
        delta_minuscula_ary = []  # el primer elem son los ultimos deltas, el segundo los anteriores y asi ....
        for i in reversed(range(len(self.layers))):  # la capa en la q toy
            delta_minuscula_ary_layer = []  # ary auxiliar para guardar los miniDelta de esta layer
            for j in reversed(range(len(self.layers[i].neurons))):  # la neurona de esa capa
                neuron = self.layers[i].neurons[j]
                if i == len(self.layers) - 1:
                    # aca asumimos una neurona al final,,,,ponele...
                    delta_minuscula = (elem[-1][0] - training_example[-1]) * neuron.activation_function.get_derivative(
                        neuron.last_activation_value)
                    delta_minuscula_ary_layer.append(delta_minuscula)
                else:
                    pesos_anteriores = []
                    for aux in range(len(self.layers[i + 1].neurons)):
                        pesos_anteriores.append(self.layers[i + 1].neurons[aux].weights[0][j])  # TODO chequear este J
                    delta_minuscula_anteriores = delta_minuscula_ary[
                        -(i + 1)]  # agarro los deltas de la capa siguiente, recordar q esta invertido
                    delta_minuscula = neuron.activation_function.get_derivative(neuron.last_activation_value) * (
                        np.dot(np.array(pesos_anteriores), np.array(delta_minuscula_anteriores)))
                    delta_minuscula_ary_layer.append(delta_minuscula)
                for wi in reversed(range(len(neuron.weights[0]) - 1)):
                    if i == 0:  # cuando esta en el principio de la red, ya no tiene que agarrar pesos sino directamente el input
                        V = training_example[wi]
                    else:
                        V = elem[i - 1][wi]

                    delta = neuron.learning_rate * delta_minuscula * V
                    neuron.weights[0][wi] += delta
            delta_minuscula_ary.append(
                delta_minuscula_ary_layer)  # agrego los miniDeltas de esta layer al ary de deltas

    # hace una ida, backprogation y repite. Realiza una epoca entera (pasar por dataset completo)
    def incremental_training(self, training_set):
        for training_example in training_set:
            ## aca voy hacia adelante
            elem = self.feed_forward(training_example)
            ## de reversa
            self.back_propagation(training_example, elem)


    def test(self, testing_set, silent = False):
        halfwaySquareError = 0
        self.error = 0
        for testing_example in testing_set:
            outputs_neuronas = self.feed_forward(testing_example)
            output = np.array(outputs_neuronas[len(outputs_neuronas)-1])
            if not silent:
                print("network answer for parameters: ", np.array_str(testing_example), " is ", np.array_str(output), " real answer is ", testing_example[-1])
            # self.error += error
            # halfwaySquareError += np.square(testing_example[-1] - output)



class Layer:
    def __init__(self, neuron_amount, connections, activation_function):  # connections seria numero de "inputs" que va a recibir (asumo que capa anterior va a estar conectada completamente)
        self.neurons = []
        for i in range(neuron_amount):
            p = Perceptron(connections, activation_function, "pending")
            self.neurons.append(p)


class LayerInfo:
    def __init__(self, activation_function, neurons_amount, connections):
        self.activation_function = activation_function
        self.neurons_amount = neurons_amount
        self.connections = connections

