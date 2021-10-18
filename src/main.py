from components.network import Network

X = [0.5, 0.6, -2.4, 5.4]
y = [1, 0, 1, 1]

nn = Network(4)
nn.add_layer(16)
nn.add_layer(2)

nn.train(X, y, 10)
