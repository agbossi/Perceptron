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

    def feed_forward(self, training_set):
        for training_example in training_set:
            ## aca voy hacia adelante
            elem = [training_example[:-1].tolist()]
            for i in range(len(self.layers)):
                elem.append(np.zeros(len(self.layers[i].neurons)).tolist())
                for j in range(len(self.layers[i].neurons)):
                    elem[i+1][j] = self.layers[i].neurons[j].run_multilayer(np.array(elem[i]))

            ## ya tengo en elem la salida de la ultima capa, ahora corrijo pesos hacia atras




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

