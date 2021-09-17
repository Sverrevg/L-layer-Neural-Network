from layer import Layer
from neuron import Neuron


class Network:
    def __init__(self, input_dim):
        # Create input layer based on input dimensions:
        self.input_layer = Layer(input_dim)
        self.hidden_layers = []
        # Create output layer, always contains 1 neuron (for binary classification at least).
        self.output_layer = Neuron()
        # Set predict mode to true in output layer:
        self.output_layer.predict = True

    # Method to add additional layers to default of input and output layer.
    def add_layer(self, neuron_count):
        layer = Layer(neuron_count)
        self.hidden_layers.append(layer)

    # Extremely simple for now, no backpropagation yet:
    def train(self, x):
        # Get outputs from input layer based on input data:
        layer_output = self.calc_layer_ouput(self.input_layer.neurons, x)
        print(layer_output)

        # Then feed this array to each neuron in next layer:
        for layer in self.hidden_layers:
            layer_output = self.calc_layer_ouput(layer.neurons, layer_output)
            print(layer_output)

        # Finally, calculate final output:
        prediction = self.output_layer.feedforward(layer_output)
        print(prediction)

    def calc_layer_ouput(self, neurons, x):
        output = []
        for neuron in neurons:
            output.append(neuron.feedforward(x))

        return output
