from components import node
import numpy as np


class NeuralNetwork:
    def __init__(self, x, y, epochs):
        self.x = x
        self.y = y
        self.epochs = epochs

        # Create empty arrays to store nodes:
        self.input_layer = []
        self.hidden_layers = []

        # First build input layer. Use input count:
        for i in range(len(self.x)):
            n = node.Node()
            self.input_layer.append(n)

    def add_layer(self, node_count):
        layer = []
        print(f"Layer {len(self.hidden_layers)} added. Node count: {node_count}")
        for i in range(node_count):
            n = node.Node()
            layer.append(n)

        self.hidden_layers.append(layer)

    def train(self):
        # Run epochs:
        for i in range(self.epochs):
            # Input nodes:
            i = 0
            for node in self.input_layer:
                output = node.feedforward(self.x[i])
                print(f"Input: {self.x[i]}, output: {output}")
                # Now forward to other layers.
                if node.is_active:
                    self.forward_layer(self.hidden_layers[0], output)
                # for layer in self.layers:
                #     self.forward_layer(layer, output)
                i += 1


    def forward_layer(self, layer, value):
        for node in layer:
            output = node.feedforward(value)
            print(f"Next layer input: {value}, output: {output}")

    def backpropagation(self, node_output, y, output):
        # Calculate loss:
        d_weights2 = np.dot(node_output, (2 * (y - output) * self.sigmoid_derivative(output)))
        # Application of the chain rule to find derivative of the loss function with respect to weights2 and weights1

        # Update the weights with the derivative (slope) of the loss function
        return 0

    # Function to calculate sigmoid.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Function to calculate the derivative of the sigmoid of x.
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
