import numpy as np


def step_function(value):
    if value > 0:
        return 1
    else:
        return 0


def sigmoid_function(value):  # los parametros van harcodeados por ahora porque me estoy haciendo un quilombo. quizas solucionable con objetos o arg variables
    z = (88.184 / (1 + np.exp(-value))) + 0.32
    return z
