import numpy as np

from src.components.layer import Layer
from src.components.neuron import Neuron


# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)


def calc_layer_output(neurons, x):
    output = []
    for neuron in neurons:
        output.append(neuron.feedforward(x))

    return output


def calc_weights(neurons, node_weights, loss):
    # Get all nodes in layer:
    for i, neuron in enumerate(neurons):
        # Then sum up all weights associated with this node:
        # Get record with index i from each array in node_weights.
        summed_weights = 0
        for array in node_weights:
            summed_weights += array[i]

        d_weights = np.dot(neuron.signal, (2 * loss * sigmoid_derivative(summed_weights)))
        neuron.update_weights(d_weights)


class Network:
    def __init__(self, input_dim):
        # Create input layer based on input dimensions:
        self.input_layer = Layer(input_dim)
        self.hidden_layers = []
        # Create output layer, always contains 1 neuron (for binary classification at least).
        self.output_layer = Neuron()
        self.loss = 0

    # Method to add additional layers to default of input and output layer.
    def add_layer(self, neuron_count):
        layer = Layer(neuron_count)
        self.hidden_layers.append(layer)

    # Extremely simple for now, no backpropagation yet:
    def train(self, x, y, iterations):
        # Run multiple iterations:
        for iteration in range(iterations):
            # Run for each supplied input:
            for index, value in enumerate(x):
                # Get outputs from input layer based on input data:
                layer_output = calc_layer_output(self.input_layer.neurons, [value])

                # Then feed this array to each neuron in next layer:
                for layer in self.hidden_layers:
                    layer_output = calc_layer_output(layer.neurons, layer_output)

                # Calculate final output:
                prediction = self.output_layer.feedforward(layer_output)

                self.calc_total_loss(y[index], prediction)

            # Then, calculate backpropagation for each layer:
            # Output layer first, using the prediction made:
            self.backpropagation(self.loss)
            print(f"Iteration {iteration}: \n loss: {self.loss}")

            # Reset loss after performing backprop:
            self.loss = 0

    def calc_total_loss(self, y, output):
        # Use square function
        loss = np.square(y - output)
        self.loss += loss

    def backpropagation(self, loss):
        # Start with output layer:
        d_weights = np.dot(self.output_layer.signal, (2 * loss * sigmoid_derivative(loss)))
        self.output_layer.update_weights(d_weights)

        node_weights = self.get_layer_weights([self.output_layer])

        # Then hidden layers:
        index = len(self.hidden_layers)

        # Reverse iterate over layers
        while index > 0:
            index -= 1
            calc_weights(self.hidden_layers[index].neurons, node_weights, loss)

            # Next, get updated weights for next layer:
            node_weights = self.get_layer_weights(self.hidden_layers[index].neurons)

    def get_layer_weights(self, neurons):
        # Get all weights for input layer and store in array. Index corresponds to index in node array:
        # Structure: Array of 'nodes' with each an array of weights. Arrays within an array.
        node_weights = []
        weights = []
        for neuron in neurons:
            for value_pair in neuron.value_pairs:
                # Create array as there can be multiple weights connected to previous nodes:
                weight = value_pair.get_weight()
                weights.append(weight)

            # Add array of weights to node_weights:
            node_weights.append(weights)

        return node_weights
