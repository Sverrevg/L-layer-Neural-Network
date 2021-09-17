import numpy as np
import random


class Node:
    is_active = True

    def __init__(self):
        self.weights = []
        self.bias = random.uniform(-1, 1)

    # Calculate output:
    def feedforward(self, input):
        return self.relu(np.dot(input, self.weight) + self.bias)

    # Activation function: output input directly if positive, otherwise, output zero.
    def relu(self, x):
        if x > 0:
            self.is_active = True
            return x
        else:
            self.is_active = False
            return 0