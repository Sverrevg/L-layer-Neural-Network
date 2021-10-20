import numpy as np
import unittest

import math_operations
import neural_net


class TestMethods(unittest.TestCase):
    def test_sigmoid(self):
        data = np.array([0, 2])
        output = math_operations.sigmoid(data)
        expected = (np.array([0.5, 0.88079708]))

        np.testing.assert_allclose(output, expected)

    def test_relu(self):
        data = np.array([-1.54, 2.23])
        output = np.array([math_operations.relu(data[0]), math_operations.relu(data[1])])
        expected = np.array([0, 2.23])

        np.testing.assert_allclose(output, expected)

    def test_sigmoid_derivative(self):
        data = np.array([0.57, 0.88])
        output = np.array([math_operations.sigmoid_derivative(data[0]), math_operations.sigmoid_derivative(data[1])])
        expected = np.array([0.2451, 0.1056])

        np.testing.assert_allclose(output, expected)

    def test_relu_derivative(self):
        data = np.array([-0.57, 0.88])
        output = np.array([math_operations.relu_derivative(data[0]), math_operations.relu_derivative(data[1])])
        expected = np.array([0, 1])

        np.testing.assert_allclose(output, expected)

    def test_nn_methods(self):
        X = np.array([[4, -7, 2, 6]])
        Y = np.array([[1, 0, 1, 1]])
        input_length = X.shape[1]

        layers = [8, 2, 1]
        parameters = neural_net.initialize_parameters(X.shape[0], layers)
        print(f"Parameters: {parameters}")

        layer_count = len(layers)
        S, Z, A = neural_net.forward_propagation(X, parameters, layer_count)
        print(f"Sigmoid output: {S}")

        cost = neural_net.compute_cost(S, Y)
        print(f"Cost: {cost}")
