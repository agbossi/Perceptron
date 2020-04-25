#1. set b = w = 0
#2. for N iterations, or until weights do not change
#       (a) for each training example xᵏ with label yᵏ
#           i. if yᵏ — f(xᵏ) = 0, continue
#           ii. else, update wᵢ, △wᵢ = (yᵏ — f(xᵏ)) xᵢ

#IMPLEMENTACION PRACTICA PERCEPTRON CATEDRA EN MATLAB
#i = 0;
#w = rand(N, 1) * 2 - 1;
#error = 1;
#errormin = p * 2;
#whileerror>0∧i<COTA
#if i<100∗n
#w = rand(N, 1) * 2 - 1;
#endix = ceil(rand(1, 1) * p);
#exitacion = x(ix,:) * w;activacion = signo(exitacion);
#∆w =α* (y(1,ix) - activacion) * x(ix,:)’;
#w = w + ∆w;error = CalcularError(x, y, w, p);
#if errorerror<errorminerrormin = error;
#wmin = w;
#endi = i + 1;
#end


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    URL_ = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    data = pd.read_csv(URL_, header=None)
    print(data)

    # make the dataset linearly separable
    data = data[:100] #hasta elem 101?
    data[4] = np.where(data.iloc[:, -1] == 'Iris-setosa', 0, 1) # toma el array, selecciona todas las filas. y la ultima columna, y aplica condicion para reemplazo sobre este subconjunto.
    data = np.asmatrix(data, dtype='float64') #queda una matriz donde la ultima columna tiene 0 o 1 segun categoria. para que quiero el dtype?
    return data


data_set = load_data()

plt.scatter(np.array(data_set[:50, 0]), np.array(data_set[:50, 2]), marker='o', label='setosa')
plt.scatter(np.array(data_set[50:, 0]), np.array(data_set[50:, 2]), marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend()
plt.show()

