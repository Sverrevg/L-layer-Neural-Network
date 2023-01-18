from enum import Enum


class Loss(Enum):
    BINARY = 'binary-cross-entropy'
    CATEGORICAL = 'categorical-cross-entropy'
