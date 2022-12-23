from unittest import TestCase

import numpy as np

from src.math_operations import sigmoid


class TestMathOperations(TestCase):
    def test_sigmoid(self):
        input_data = np.array([2.3, 55.5, 654.4])
        expected = np.array([0.90887704, 1., 1.])

        output, _ = sigmoid(input_data)

        for i in range(input_data.shape[0]):
            # Use almostEquals as the function returns a float:
            self.assertAlmostEqual(output[i], expected[i])
