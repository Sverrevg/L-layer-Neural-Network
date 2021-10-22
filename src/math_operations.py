import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x)), x


def relu(x):
    g = np.array(x)
    g[g < 0] = 0

    return g, x


def relu_derivative(x):
    return 1 if x >= 0 else 0


def relu_backward(A, Z):
    return np.multiply(A, relu_derivative(Z))


def sigmoid_derivative(x):
    return np.multiply(x, (1 - x))


def sigmoid_backward(A, Z):
    return np.multiply(A, sigmoid_derivative(Z))
