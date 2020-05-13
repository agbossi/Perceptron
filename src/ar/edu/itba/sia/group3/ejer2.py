import numpy as np
# TODO hacer menos canceroso el codigo

import ar.edu.itba.sia.group3.Metrics.Graphics as mtr
import ar.edu.itba.sia.group3.Models.Perceptron_neuron as md  # no puedo traer el paquete?
import ar.edu.itba.sia.group3.Functions.Activation_Functions as af
import src.ar.edu.itba.sia.group3.Resamplers.Train_test_split as rs
import src.ar.edu.itba.sia.group3.Resamplers.K_fold_cross_validation as crossValidation
import pandas as pd
from os.path import expanduser as ospath
from sklearn.utils import shuffle

# ospath('~/PycharmProjects/Perceptron/TP3-ej2-Conjunto_entrenamiento.xlsx')
def load_data_set():
    data = pd.read_excel('TP3-ej2-Conjunto_entrenamiento.xlsx').iloc[1:, :4]  # TODO manejar path de archivo
    data = shuffle(data)
    return data.to_numpy()

activation_function = af.SigmoidFunction(0.5)
features = 3
iteration_limit = 70
restart_condition = 5
learning_rate = 0.2

df = load_data_set()

input_list_of_list = crossValidation.cross_validation_split(5, df)
# input_list_of_list = rs.train_test_split(df, 0.7)

results = []
for input_list in input_list_of_list:
    p = md.Perceptron(features, activation_function, "regression")
    trained_weights, errors_per_epoch = p.batch_training(input_list[0],
                                                         learning_rate, restart_condition, iteration_limit, True)
    # mtr.converge_metric(iteration_limit, errors_per_epoch)
    error, sqrError = p.test_perceptron(input_list[1], True)
    results.append([trained_weights, errors_per_epoch, sqrError])

results = np.array(results)

exit()
# TODO tratar de graficar todos los resultados en unico grafico o algo asi?

# TODO preguntar si esta todo ok usar numpy y sklearn al estar usando cosas triviales

# TODO add cross validation --> ya nos funciona ahora q mierda hacemos?

# TODO add bootstraping --> esto hay que hacerlo? muy poco contenido. pedir que lo expliquen de nuevo?

# PREGUNTAS
#¿Co ́mo podr ́ıa escoger el mejor conjunto de entrenamiento?
# Hacer cross validation para armar multiples splits de la data y probarlos, el split que te de mejores resultados
# para su conjunto de testeo se podria decir que es el mejor conjunto pero podria pasar que justo solo sirva para ese
# conjunto de testeo de ese split.
# Buscar otra forma de encontrar el mejor?

# ¿Co ́mo podr ́ıa evaluar la m ́axima capacidad de generalizaci ́on del perceptron para este conjunto de datos?
# TODO responder


# Tratar de responder las preguntas del ppt
#  ¿Co ́mo sabemos si la divisi ́on en conjunto de entrenamiento y conjunto de prueba es apropiada?
# ¿Co ́mo evaluamos cuantitativamente la capacidad de clasificaci ́on?
# ¿Es verdad que si aumento la cantidad de  ́epocas, entonces el m ́etodo clasifica mejor?
# ¿Es verdad que si E(w) ≡ 0, entonces es un clasificador perfecto?