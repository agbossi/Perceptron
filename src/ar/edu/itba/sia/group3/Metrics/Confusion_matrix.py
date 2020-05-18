import enum


class MatrixComponents(enum.Enum):
    true_positive = 0
    false_positive = 1
    true_negative = 2
    false_negative = 3


class ConfusionMatrix:
    def __init__(self, possible_classifications):
        self.matrix = [[int(0) for i in range(len(possible_classifications))] for j in range(len(possible_classifications))]
        self.stats_matrix = None
        self.classifications = possible_classifications
        self.entries = 0

    def add_entry(self, real_classification, classification):
        self.matrix[real_classification][classification] += 1
        self.entries += 1

    def summarize(self):
        stats_matrix = [[int(0) for j in range(4)] for i in range(len(self.classifications))]
        for k in range(len(self.classifications)):
            curr_classification_amount = 0
            stats_matrix[k][MatrixComponents.true_positive.value] += self.matrix[k][k]
            for l in range(len(self.classifications)):
                if k != l:
                    stats_matrix[k][MatrixComponents.false_positive.value] += self.matrix[l][k]
                    stats_matrix[k][MatrixComponents.false_negative.value] += self.matrix[k][l]
                    curr_classification_amount += (self.matrix[k][l]+self.matrix[l][k]+self.matrix[k][k])
            stats_matrix[k][MatrixComponents.false_negative.value] += curr_classification_amount - stats_matrix[k][MatrixComponents.true_positive.value] - stats_matrix[k][MatrixComponents.false_positive.value] - stats_matrix[k][MatrixComponents.false_negative.value]
            stats_matrix[k].insert(0, self.classifications[k].name)
        stats_matrix.insert(0, [" ", "TP", "FP", "TN", "FN"])
        self.stats_matrix = stats_matrix
        return stats_matrix

    def print_confusion_matrix(self):
        header = []
        for k in range(len(self.classifications)):
            self.matrix[k].insert(0, self.classifications[k].name)
            header.append(self.classifications[k].name)
        header.insert(0, " ")
        self.matrix.insert(0, header)
        print_m(self.matrix)

    def print_summary(self):
        print_m(self.stats_matrix)

    def get_precisions(self):
        precisions = []
        if self.stats_matrix is None:
            self.summarize()
            for i in range(len(self.classifications)):
                precision = [self.classifications[i].name, self.get_precision(i+1)]  # paso la linea con texto
                precisions.append(precision)
        return precisions

    def get_precision(self, index):
        precision = ((self.stats_matrix[index][MatrixComponents.true_positive.value] + self.stats_matrix[index][MatrixComponents.true_negative.value])
                     / (self.stats_matrix[index][MatrixComponents.true_positive.value] + self.stats_matrix[index][MatrixComponents.true_negative.value] + self.stats_matrix[index][MatrixComponents.false_negative.value] + self.stats_matrix[index][MatrixComponents.false_positive.value]))
        return precision

    def get_accuracies(self):
        if self.stats_matrix is None:
            self.summarize()
        accuracies = []
        for i in range(len(self.classifications)):
            accuracy = [self.classifications[i].name, self.get_accuracy(i + 1)]  # paso la linea con texto
            accuracies.append(accuracy)
        return accuracies

    def get_accuracy(self, index):
        accuracy = self.stats_matrix[index][MatrixComponents.true_positive.value] / (self.stats_matrix[index][MatrixComponents.true_positive.value] + self.stats_matrix[index][MatrixComponents.false_positive.value])
        return accuracy

    def get_recalls(self):
        if self.stats_matrix is None:
            self.summarize()
        recalls = []
        for i in range(len(self.classifications)):
            recall = [self.classifications[i].name, self.get_recall(i + 1)]  # paso la linea con texto
            recalls.append(recall)
        return recalls

    def get_recall(self, index):
        recall = self.stats_matrix[index][MatrixComponents.true_positive.value] / (self.stats_matrix[index][MatrixComponents.true_positive.value] + self.stats_matrix[index][MatrixComponents.false_negative.value])
        return recall

    def get_f1_scores(self):
        if self.stats_matrix is None:
            self.summarize()
        f1_scores = []
        for i in range(len(self.classifications)):
            f1_score = [self.classifications[i].name, self.get_f1_score(i + 1)]  # paso la linea con texto
            f1_scores.append(f1_score)
        return f1_scores

    def get_f1_score(self, index):
        f1_score = 2 * self.get_precision(index) * self.get_recall(index) / (self.get_recall(index) + self.get_precision(index))
        return f1_score

def print_m(matrix):
    for i in range(len(matrix[0])):
        for j in range(len(matrix)):
            print(matrix[i][j], end=" ")
        print('\n')
