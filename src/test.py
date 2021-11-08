from neural_net import NeuralNetwork
import numpy as np

layers_dims = [3, 6, 4]

nn = NeuralNetwork(layers_dims=layers_dims, activation="softmax")

X = np.array([[2.3, 4.5, -3.4, 6.7, 1.4],
              [5.4, -2.7, 1.9, -0.8, 4.3],
              [3.5, 6.7, 8.2, -9.1, 2.5]])
y = np.array([[1, 0, 2, 3, 2]])

nn.fit(X, y)
