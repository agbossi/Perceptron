import numpy as np


def step_function(value):
    if value > 0:
        return 1
    else:
        return 0


class StepFunction:
    def get_value(self, value):
        if value > 0:
            return 1
        else:
            return 0

    def get_derivative(self, value):
        return np.array([1])  # esto casi seguro esta mal


class SigmoidFunction:
    def __init__(self, beta):
        self.beta = beta

    def get_value(self, value):
        z = 1 / (1 + np.exp((-2 * self.beta * value)))
        return z

    def get_derivative(self, value):
        z = 2 * self.beta * self.get_value(value) * (1 - self.get_value(value))
        return z


def ReLU(value):
    return max(0, value)
