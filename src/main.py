import NeuralNetwork
import numpy as np

learning_rate = 0.1
input_vector = np.array([2, 1.5])

neural_network = NeuralNetwork.NeuralNetwork(learning_rate)

neural_network.predict(input_vector)
