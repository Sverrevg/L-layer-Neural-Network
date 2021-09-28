from components.network import Network

inputs = [0.5, 0.6, -2.4, 5.4]

nn = Network(4)
nn.add_layer(16)
nn.add_layer(2)

nn.train(inputs, 1, 10)
