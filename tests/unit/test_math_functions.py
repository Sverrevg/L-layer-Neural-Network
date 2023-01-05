from unittest import TestCase

import numpy as np

from neural_network.network_functions.math_functions import sigmoid, relu, softmax, relu_backward, sigmoid_backward, softmax_backward


class MathFunctionsTestSuite(TestCase):
    def setUp(self) -> None:
        self.input_data = np.array([0.07050, 0.07061, -0.06694])
        self.cache = np.array([0.27520, 0.24979, 0.37615])

    def test_sigmoid(self) -> None:
        expected = np.array([0.51762, 0.51762, 0.48327])
        output, _ = sigmoid(self.input_data)

        self.assertEqual(expected.all(), output.all())

    def test_relu(self) -> None:
        expected = np.array([0.07050, 0.07061, 0.])
        output, _ = relu(self.input_data)

        self.assertEqual(expected.all(), output.all())

    def test_softmax(self) -> None:
        expected = np.array([0.34823, 0.34826, 0.30351])
        output, _ = softmax(self.input_data)

        self.assertEqual(expected.all(), output.all())

    def test_relu_backward(self) -> None:
        expected = np.array([0.07050, 0.07061, -0.06694])
        output = relu_backward(self.input_data, self.cache)

        self.assertEqual(expected.all(), output.all())

    def test_sigmoid_backward(self) -> None:
        expected = np.array([0.01730, 0.01738, -0.01616])
        output = sigmoid_backward(self.input_data, self.cache)

        self.assertEqual(expected.all(), output.all())

    def test_softmax_backward(self) -> None:
        expected = np.array([0.01730, 0.01738, -0.01616])
        output = softmax_backward(self.input_data, self.cache)

        self.assertEqual(expected.all(), output.all())
