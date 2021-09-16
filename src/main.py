from src import neural_network

input_data = [5, 7, 2.3]
labels = [0, 1, 0]

nn = neural_network.NeuralNetwork([1.89, 4.56], [0, 1], 1)

# Add two hidden layers:
nn.add_layer(4)
nn.add_layer(2)
nn.train()