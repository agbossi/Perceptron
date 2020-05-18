import enum

import numpy as np
import ar.edu.itba.sia.group3.Models.Multi_layer_perceptron as mdL
import ar.edu.itba.sia.group3.Functions.Activation_Functions as af


xor_function = [['x', 'y', 'or'], [1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 1, 1]]
xor_data_set = np.array(xor_function[1:])

activation_function = af.SigmoidFunction(2)


class ClassificationTypes(enum.Enum):
    cero = 0
    uno = 1


classifications = [ClassificationTypes.cero, ClassificationTypes.uno]

features = 2
iteration_limit = 5000
learning_rate = 0.5
momentum = 0.5  # cero to disable momentum
## Set up layers for neural network
layer_info_list = [
    mdL.LayerInfo(activation_function, 3, 2),
    mdL.LayerInfo(activation_function, 1, 3)
]

## Create and runs neural network
p = mdL.MultiLayerPerceptron(2, layer_info_list, 1, learning_rate, momentum)
p.incremental_training(xor_data_set, iteration_limit)
confusion = p.test_classification(xor_data_set, classifications)
print(confusion.get_accuracies())

