from enum import Enum


class Activation(Enum):
    SIGMOID = 'sigmoid'
    RELU = 'relu'
    SOFTMAX = 'softmax'


class Loss(Enum):
    BINARY = 'binary-cross-entropy'
    CATEGORICAL = 'categorical-cross-entropy'


class Optimizer(Enum):
    SGD = 'stochastic-gradient-descent'
    SGDM = 'stochastic-momentum'
