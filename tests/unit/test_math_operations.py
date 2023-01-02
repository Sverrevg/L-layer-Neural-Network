from unittest import TestCase

import numpy as np

from neural_network.math_operations import sigmoid, relu, softmax


class TestMathOperations(TestCase):
    def setUp(self) -> None:
        self.input_data = np.array([2.3, 55.5, 654.4, -1.6])

    def test_sigmoid(self) -> None:
        expected = np.array([0.9089, 1., 1., 0.168])

        output, _ = sigmoid(self.input_data)

        for i in range(self.input_data.shape[0]):
            self.assertEqual(expected[i], np.round(output[i], 4))

    def test_relu(self) -> None:
        expected = np.array([2.3, 55.5, 654.4, 0.])

        output, _ = relu(self.input_data)

        for i in range(self.input_data.shape[0]):
            self.assertEqual(expected[i], output[i])

    def test_softmax(self) -> None:
        expected = np.array([0, 0, 1, 0])

        output, _ = softmax(self.input_data)

        for i in range(self.input_data.shape[0]):
            self.assertEqual(expected[i], np.round(output[i], 4))

    # def test_relu_backward(self) -> None:
    #     expected = np.array(0)

        # output = relu_backward(self.input_data, )
