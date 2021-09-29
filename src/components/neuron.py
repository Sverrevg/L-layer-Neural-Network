import random
import numpy as np

from src.components.value_pair import ValuePair


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# The neuron takes all inputs, each with their own weights, and calculates the weighted sum.
# Then, after adding the bias, if the output > t activate output.
class Neuron:
    def __init__(self):
        self.inputs = []
        self.value_pairs = []
        self.signal = 0
        # The higher the learning rate, the faster the weights and bias change.
        # Start with higher value and then gradually decrease?
        self.learning_rate = 0.1
        self.bias = random.uniform(-1, 1)
        self.predict = False

    # This method sets the weights for each input provided.
    def set_weights_values(self, inputs):
        for raw_input in inputs:
            self.value_pairs.append(ValuePair(random.uniform(-1, 1), raw_input))

    # This method gets all inputs, multiplies those with the assigned weight and then sums the outcomes.
    def feedforward(self, inputs):
        self.set_weights_values(inputs)
        weighted_sum = 0
        for value_pair in self.value_pairs:
            # Calculate weighted input and add to weighted sums:
            weighted_sum += value_pair.get_weight() * value_pair.get_raw_input()

        self.signal = sigmoid(weighted_sum + self.bias)
        return self.signal

    # Function to calculate sigmoid.

    # For all inputs:
    # weight = weight + (learning_Rate * input * diff)
    # And:
    # bias = bias + (lr * diff)
    def update_weights(self, diff):
        # Update the weights in each value pair:
        for value_pair in self.value_pairs:
            value_pair.weight += (self.learning_rate * value_pair.get_raw_input * diff)

        # Update bias for this neuron:
        self.bias += (self.learning_rate * diff)
