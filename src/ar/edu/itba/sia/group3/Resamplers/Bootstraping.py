import numpy as np
from sklearn.utils import shuffle


def bootstraping(data_set, m):
    data = data_set.iloc[1:, :4]
    data = shuffle(data).to_numpy()
    element_set = []
    for i in range(m):
        element = np.empty(shape=(len(data), len(data[0])))
        for j in range(len(data)):
            element = np.vstack(element, data[np.random.rand(1, 1), :])
        element_set.append(element) #faltaria hacer la lista contenedora con su respectivo ts
    return element_set
