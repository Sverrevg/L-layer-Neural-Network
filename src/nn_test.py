import numpy as np
import unittest

import neural_net


class TestMethods(unittest.TestCase):
    def test_sigmoid(self):
        data = np.array([0, 2])
        output = neural_net.sigmoid(data)
        expected = (np.array([0.5, 0.88079708]))

        np.testing.assert_allclose(output, expected)

    def test_relu(self):
        data = np.array([-1.54, 2.23])
        output = np.array([neural_net.relu(data[0]), neural_net.relu(data[1])])
        expected = np.array([0, 2.23])

        np.testing.assert_allclose(output, expected)

    def test_sigmoid_derivative(self):
        data = np.array([0.57, 0.88])
        output = np.array([neural_net.sigmoid_derivative(data[0]), neural_net.sigmoid_derivative(data[1])])
        expected = np.array([0.2451, 0.1056])

        np.testing.assert_allclose(output, expected)

    def test_relu_derivative(self):
        data = np.array([-0.57, 0.88])
        output = np.array([neural_net.relu_derivative(data[0]), neural_net.relu_derivative(data[1])])
        expected = np.array([0, 1])

        np.testing.assert_allclose(output, expected)
