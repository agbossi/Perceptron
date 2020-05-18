import enum

import numpy as np
import ar.edu.itba.sia.group3.Models.Multi_layer_perceptron as mdL
import ar.edu.itba.sia.group3.Functions.Activation_Functions as af
import ar.edu.itba.sia.group3.Resamplers.K_fold_cross_validation as crossvalidation

primos_data_set = \
    [[0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
     [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
     [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1],
     [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
     [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1],
     [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
     [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0]
     ]
primos_data_set = np.array(primos_data_set)

activation_function = af.SigmoidFunction(.2)

features = 35
iteration_limit = 6000
learning_rate = 0.3
momentum = 0.2  # cero to disable momentum
## Set up layers for neural network
layer_info_list = [
    mdL.LayerInfo(activation_function, 23, 35),
    mdL.LayerInfo(activation_function, 1, 23)
]


class ClassificationTypes(enum.Enum):
    primo = 0
    no_primo = 1


classifications = [ClassificationTypes.primo, ClassificationTypes.no_primo]

input_list_of_list = crossvalidation.cross_validation_split(2, primos_data_set)

results = []
for input_list in input_list_of_list:
    ## Create and runs neural network
    p = mdL.MultiLayerPerceptron(features, layer_info_list, 1, learning_rate, momentum)
    p.incremental_training(input_list[0], iteration_limit)
    confusion_matrix = p.test_classification(input_list[1], classifications,True)
    confusion_matrix.print_confusion_matrix()
    results.append([input_list[1], confusion_matrix.get_accuracies()])



split_num = 1
for result in results:
    print("Network ", split_num, " accuracies", result[1])
    split_num += 1


