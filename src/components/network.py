import numpy as np

from components.layer import Layer
from components.neuron import Neuron


def sigmoid_derivative(x):
    return x * (1 - x)


def calculate_loss(y, output):
    # Use square function
    return (y - output) * (y - output)


def calc_layer_output(neurons, x):
    output = []
    for neuron in neurons:
        output.append(neuron.feedforward(x))

    return output


def backpropagation(neuron, loss):
    # Application of the chain rule to find derivative of the loss function with respect to weights
    d_weights = np.dot(neuron.signal, (2 * loss * sigmoid_derivative(loss)))

    neuron.update_weights(d_weights)


class Network:
    def __init__(self, input_dim):
        # Create input layer based on input dimensions:
        self.input_layer = Layer(input_dim)
        self.hidden_layers = []
        # Create output layer, always contains 1 neuron (for binary classification at least).
        self.output_layer = Neuron()

    # Method to add additional layers to default of input and output layer.
    def add_layer(self, neuron_count):
        layer = Layer(neuron_count)
        self.hidden_layers.append(layer)

    # Extremely simple for now, no backpropagation yet:
    def train(self, x, y, iterations):
        for i in range(iterations):
            # Get outputs from input layer based on input data:
            layer_output = calc_layer_output(self.input_layer.neurons, x)

            # Then feed this array to each neuron in next layer:
            for layer in self.hidden_layers:
                layer_output = calc_layer_output(layer.neurons, layer_output)

            # Calculate final output:
            prediction = self.output_layer.feedforward(layer_output)
            print(prediction)

            # # Then, calculate backpropagation for each layer:
            # # Output layer first, using the prediction made:
            # loss = calculate_loss(y, prediction)
            # backpropagation(self.output_layer, loss)
            #
            # # Next, hidden layers:
            # for layer in self.hidden_layers:
            #     for neuron in layer.neurons:

    # Derivative of sigmoid
