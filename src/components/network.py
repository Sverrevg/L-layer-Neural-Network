import numpy as np


class NeuralNetwork:
    def __init__(self, input_count, x, y, epochs):
        self.input_count = input_count
        self.input = x
        self.y = y
        self.epochs = epochs
        self.input_layer = []
        self.layers = [[]]
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.output = np.zeros(y.shape)

    def add_layer(self, node_count):
        self.layers.append(node_count)

    # Perform the next step in the sequence.
    # This represents one single node with only two layers.
    def feedforward(self):
        # Calculate output value for each node in input layer using relu activation:
        for i in range(self.input_count):
            self.input_layer.append(self.relu(np.dot(self.input, self.weights1)))

        # Now for the layers:
        for layer in self.layers:
            for i in layer:
                print(i)

        self.output = self.sigmoid(np.dot(self.node, self.weights2))

    # Calculate loss and perform backpropagation.
    def backpropagation(self, node):
        # Application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(node.T, (2 * (self.y - self.output) * self.sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * self.sigmoid_derivative(self.output),
                                                  self.weights2.T) * self.sigmoid_derivative(node)))

        # Update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    # Activation function: output input directly if positive, otherwise, output zero.
    def relu(self, x):
        if x > 0:
            return x
        else:
            return 0

    # Function to calculate sigmoid.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Function to calculate the derivative of the sigmoid of x.
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


nn = NeuralNetwork(8, 12, 12, 12)
nn.add_layer(32)
nn.feedforward()
