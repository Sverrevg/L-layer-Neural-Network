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
        data = [-1.54, 2.23]
        output = [neural_net.relu(data[0]), neural_net.relu(data[1])]
        expected = [0, 2.23]

        np.testing.assert_allclose(output, expected)
