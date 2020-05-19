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
# activation_function = af.StepFunction()

features = 35
iteration_limit = 100
learning_rate = 0.9
momentum = 0  # cero to disable momentum
## Set up layers for neural network
layer_info_list = [
    mdL.LayerInfo(activation_function, 23, 35),
    mdL.LayerInfo(activation_function, 1, 23)
]


class ClassificationTypes(enum.Enum):
    primo = 0
    no_primo = 1


classifications = [ClassificationTypes.primo,ClassificationTypes.no_primo]

input_list_of_list = crossvalidation.cross_validation_split(3, primos_data_set)

results = []
for input_list in input_list_of_list:
    ## Create and runs neural network
    p = mdL.MultiLayerPerceptron(features, layer_info_list, 1, learning_rate, momentum)
    p.incremental_training(primos_data_set, iteration_limit)
    confusion_matrix = p.test_classification(primos_data_set, classifications)
    results.append([input_list[1], confusion_matrix.get_accuracies()])

results = np.array(results)

split_num = 1
for result in results:
   print("Network ", split_num," accuracies",result[split_num])
   split_num +=1


exit()
input_list_of_list = crossvalidation.cross_validation_split(3, primos_data_set)

results = []
for input_list in input_list_of_list:
   p = md.Perceptron(features, activation_function, "regression")
   trained_weights, errors_per_epoch = p.batch_training(input_list[0],
                                                        learning_rate, restart_condition, iteration_limit, True)
   # mtr.converge_metric(iteration_limit, errors_per_epoch)
   error, sqrError = p.test_perceptron(input_list[1], True)
   results.append([input_list[1], errors_per_epoch, sqrError])

results = np.array(results)

split_num = 1
for result in results:
   print("neuron answer for split ", split_num, " has ",result[2]," square error")
   split_num +=1

