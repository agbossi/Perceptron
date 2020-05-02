# returns a list? with one training set and one testing set
def train_test_split(data_set, training_percent):
    sets = []

    training_set = data_set[0:int(len(data_set) * training_percent), :]
    testing_set = data_set[int(len(data_set) * training_percent):len(data_set), :]

    element_set = [training_set, testing_set]
    sets.append(element_set)
    return sets
