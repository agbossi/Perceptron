# returns a list of list of k-1 training sets and one testing set. if k = 1 returns classic test - train
from ar.edu.itba.sia.group3.Resamplers.Train_test_split import train_test_split
import numpy as np


def cross_validation_split(k, data_set):
    if k == 1:
        return train_test_split(data_set, 0.5) # if K=1 safe by splitting halfway

    # To handle case data_set len is not divisible by K, assume data set len is smaller
    # as necessary as to not have remainder on division and making it divisible.
    # operate assuming that len. Then, after all operations, grab the substracted elements from the array and
    # add them to the final sets
    aux_data_set_len = len(data_set) - (len(data_set) % k)

    # fold size is the size of every split
    fold_size = int(aux_data_set_len / k)

    # the list of splits to populate and return
    fold_set = []
    for i in range(k):  # si ds es por ejemplo 10, k es 3, hay un dato que me queda fuera de todos, me lo meto en orto?
        training_set = np.array([np.zeros(len(data_set[0])).tolist()])  # para que concatenate no explote, no encontre algo mejor
        for j in range(k):
            # con esto hacemos la diagonal que hay en el PPT haciendo que saltea el de la diagonal pq
            # sera el de testeo y el resto sera training en cada iter
            if i != j:
                training_set = np.concatenate((training_set, data_set[j*fold_size:(j+1)*fold_size, :]), 0) #aca agarro el de entrenamiento
        fold = [training_set[1:, :], data_set[i*fold_size:(i+1)*fold_size, :]] #aca agarro el de testeo q estaria en i=j
        fold_set.append(fold)

    # now we have to add any removed elements if data_set len / K had remainder
    if (len(data_set) % k) > 0:
        number_of_elements_to_add = len(data_set) % k
        for i in range(k):
            # agrego en la posicion 0 el elemento data_set[aux_data_set_len+i]
            selected_split = fold_set[i][0]
            to_concat = np.array([data_set[aux_data_set_len+i, :].tolist()])
            fold_set[i][0] = np.concatenate((selected_split, to_concat), 0)
            # fold_set[i][0] = np.concatenate(fold_set[i][0], data_set[aux_data_set_len+i, :], 0)
            # si number_of_elements_to_add == i significa que ya agregue todos los extra, hacer break
            if aux_data_set_len+i == len(data_set)-1:
                break

    return fold_set


