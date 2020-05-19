import numpy as np

import ar.edu.itba.sia.group3.Models.Perceptron_neuron as md
import ar.edu.itba.sia.group3.Functions.Activation_Functions as af
import ar.edu.itba.sia.group3.Resamplers.K_fold_cross_validation as crossValidation
import pandas as pd
from sklearn.utils import shuffle


def load_data_set():
    data = pd.read_excel('TP3-ej2-Conjunto_entrenamiento.xlsx').iloc[1:, :4]
    data = shuffle(data)
    return data.to_numpy()


activation_function = af.SigmoidFunction(0.5)
features = 3
iteration_limit = 70
restart_condition = 5
learning_rate = 0.2

df = load_data_set()

input_list_of_list = crossValidation.cross_validation_split(2, df)
# input_list_of_list = rs.train_test_split(df, 0.7)

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

