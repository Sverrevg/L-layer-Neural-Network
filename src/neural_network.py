from components import node


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

    def backpropagation(self, y):
        # Logic here
        return 0
