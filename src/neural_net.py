import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    if x > 0:
        return x
    else:
        return 0
