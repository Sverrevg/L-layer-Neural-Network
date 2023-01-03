from unittest import TestCase

from neural_network.network_operations.network_functions import initialize_parameters_deep


class NetworkFunctionsTestSuite(TestCase):
    def test_initialize_parameters(self):
        layer_dims = [10, 6, 1]
        parameters = initialize_parameters_deep(layer_dims)

        self.assertEqual(len(parameters), 4)  # 2 weights and bias arrays.
        self.assertEqual(parameters['Weights1'].size, 60)
        self.assertEqual(parameters['Weights2'].size, 6)
        self.assertEqual(parameters['bias1'].size, 6)
        self.assertEqual(parameters['bias2'].size, 1)
