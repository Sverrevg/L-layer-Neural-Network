import numpy as np
import unittest

from old import neural_net, math_operations


class TestMethods(unittest.TestCase):
    def test_sigmoid(self):
        data = np.array([0, 2])
        output, cache = math_operations.sigmoid(data)
        expected = (np.array([0.5, 0.88079708]))

        np.testing.assert_allclose(output, expected)
        np.testing.assert_allclose(data, cache)

    def test_relu(self):
        data = np.array([-1.54, 2.23, -1])
        output, cache = math_operations.relu(data)
        expected = np.array([0, 2.23, 0])

        np.testing.assert_allclose(output, expected)
        np.testing.assert_allclose(data, cache)

    def test_initialize_parameters(self):
        # Each number represents the number of nodes in one layer.
        layer_dims = [5, 4, 3]
        expected = np.array([[0.01788628, 0.0043651, 0.00096497, -0.01863493, -0.00277388],
                             [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
                             [-0.01313865, 0.00884622, 0.00881318, 0.01709573, 0.00050034],
                             [-0.00404677, -0.0054536, -0.01546477, 0.00982367, -0.01101068]])
        parameters = neural_net.initialize_parameters(layer_dims)
        W1 = np.array(parameters['W1'])

        np.testing.assert_equal(W1.shape, expected.shape)

    def test_linear_forward(self):
        expected = np.array([[3.26295337, - 1.23429987]])
        t_A = np.array([[1.62434536, - 0.61175641],
                        [-0.52817175, - 1.07296862],
                        [0.86540763, - 2.3015387]])
        t_W = np.array([[1.74481176, -0.7612069, 0.3190391]])
        t_b = np.array([[-0.24937038]])

        t_Z, cache = neural_net.linear_forward(t_A, t_W, t_b)

        np.testing.assert_allclose(t_Z, expected)

    def test_linear_activation_forward(self):
        expected_sigmoid = np.array([[0.96890023, 0.11013289]])
        expected_relu = np.array([[3.43896131, 0.]])
        t_A_prev = np.array([[-0.41675785, -0.05626683],
                             [-2.1361961, 1.64027081],
                             [-1.79343559, -0.84174737]])
        t_W = np.array([[0.50288142, -1.24528809, -1.05795222]])
        t_b = np.array([[-0.90900761]])

        output_sigmoid = neural_net.linear_activation_forward(t_A_prev, t_W, t_b, activation="sigmoid")
        output_relu = neural_net.linear_activation_forward(t_A_prev, t_W, t_b, activation="relu")

        np.testing.assert_allclose(output_sigmoid[0], expected_sigmoid)
        np.testing.assert_allclose(output_relu[0], expected_relu)

    def test_compute_cost(self):
        expected = 0.2797765635793422
        t_Y = np.array([[1, 1, 0]])
        t_AL = np.array([[0.8, 0.9, 0.4]])

        output = neural_net.compute_cost(t_AL, t_Y)

        np.testing.assert_equal(output, expected)

    def test_linear_backward(self):
        t_dZ = np.array([[1.62434536, -0.61175641, -0.52817175, -1.07296862],
                         [0.86540763, -2.3015387, 1.74481176, -0.7612069],
                         [0.3190391, -0.24937038, 1.46210794, -2.06014071]])
        array_1 = np.array([[-0.3224172, -0.38405435, 1.13376944, -1.09989127],
                            [-0.17242821, -0.87785842, 0.04221375, 0.58281521],
                            [-1.10061918, 1.14472371, 0.90159072, 0.50249434],
                            [0.90085595, -0.68372786, -0.12289023, -0.93576943],
                            [-0.26788808, 0.53035547, -0.69166075, -0.39675353]])
        array_2 = np.array([[-0.6871727, -0.84520564, -0.67124613, -0.0126646, -1.11731035],
                            [0.2344157, 1.65980218, 0.74204416, -0.19183555, -0.88762896],
                            [-0.74715829, 1.6924546, 0.05080775, -0.63699565, 0.19091548]])
        array_3 = np.array([[2.10025514],
                            [0.12015895],
                            [0.61720311]])

        expected = np.array([[-1.15171336, 0.06718465, -0.32046959, 2.09812711],
                             [0.6034588, -3.72508703, 5.81700741, -3.84326836],
                             [-0.4319552, -1.30987418, 1.72354703, 0.05070578],
                             [-0.38981415, 0.60811244, -1.25938424, 1.47191593],
                             [-2.52214925, 2.67882551, -0.67947465, 1.48119548]])

        t_dA_prev, t_dW, t_db = neural_net.linear_backward(t_dZ, [array_1, array_2, array_3])

        np.testing.assert_allclose(t_dA_prev, expected)
