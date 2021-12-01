from enum import Enum


class Activations(Enum):
    SIGMOID = 'sigmoid'
    RELU = 'relu'
    SOFTMAX = 'softmax'


class Loss(Enum):
    BINARY = 'binary-cross-entropy'
    CATEGORICAL = 'categorical-cross-entropy'
