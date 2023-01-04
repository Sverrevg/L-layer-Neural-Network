from unittest import TestCase

import numpy as np

from neural_network.helpers.activation_cache import ActivationCache
from neural_network.helpers.forward_cache import ForwardCache
from neural_network.math_functions import Array
from neural_network.network_operations.network_functions import initialize_parameters_deep, linear_forward, \
    linear_activation_forward, l_model_forward, compute_cost, linear_activation_backward, l_model_backward


class NetworkFunctionsTestSuite(TestCase):
    def setUp(self) -> None:
        self.layer_dims = [4, 2, 1]

    def test_initialize_parameters(self):
        parameters = initialize_parameters_deep(self.layer_dims)

        self.assertEqual(len(parameters), 4)  # 2 weights and bias arrays.
        self.assertEqual(parameters['Weights1'].size, 8)
        self.assertEqual(parameters['Weights2'].size, 2)
        self.assertEqual(parameters['bias1'].size, 2)
        self.assertEqual(parameters['bias2'].size, 1)

    def test_linear_activation_forward(self):
        parameters = initialize_parameters_deep(self.layer_dims)
        activations = np.array([1.4, 2.4, 5, -1.54])
        weights = parameters['Weights1']
        bias = parameters['bias1']

        outputs, cache = linear_activation_forward(activations_prev=activations,
                                                   weights=weights,
                                                   bias=bias,
                                                   activation='relu')

        self.assertEqual(outputs.shape, (2, 2))
        self.assertTrue(isinstance(cache, ActivationCache))

        outputs, cache = linear_activation_forward(activations_prev=activations,
                                                   weights=weights,
                                                   bias=bias,
                                                   activation='sigmoid')

        self.assertEqual(outputs.shape, (2, 2))
        self.assertTrue(isinstance(cache, ActivationCache))

        outputs, cache = linear_activation_forward(activations_prev=activations,
                                                   weights=weights,
                                                   bias=bias,
                                                   activation='softmax')

        self.assertEqual(outputs.shape, (2, 2))
        self.assertTrue(isinstance(cache, ActivationCache))

    def test_l_model_forward(self):
        parameters = initialize_parameters_deep(self.layer_dims)
        input_data = np.array([1.4, 2.4, 5, -1.54])
        last_activation_value, caches = l_model_forward(input_data=input_data,
                                                        parameters=parameters,
                                                        activation='sigmoid',
                                                        output_shape=1)

        self.assertEqual(last_activation_value.shape, (1, 2))
        for cache in caches:
            self.assertTrue(isinstance(cache, ActivationCache))

    def test_compute_cost(self):
        probability_vector = np.array([[0.5, 0.9, 0.4, 0.2]])
        labels = np.array([[1, 0, 0, 0]])
        expected = 0.173

        cost = compute_cost(probability_vector=probability_vector,
                            labels=labels,
                            loss='categorical-cross-entropy')

        self.assertEqual(expected, np.round(cost, 3))

    def test_l_model_backward(self):
        probability_vector = np.array([[0.5, 0.9, 0.4, 0.2]])
        labels = np.array([[1, 0, 0, 0]])
        mock_data = np.array([[1.4, 2.4, 5, -1.54],
                              [0.2, 5.3, 6.1, 1.3],
                              [3.5, 1.3, 1.3, 5.0],
                              [-0.2, 3.5, 1.7, -0.5]])
        forward_cache = ForwardCache(activations=mock_data,
                                     weights=mock_data,
                                     bias=np.array([[0.1], [-0.5], [0.6], [1.7]]))
        activation_cache = ActivationCache(forward_cache, mock_data)
        caches = [activation_cache, activation_cache]

        grads = l_model_backward(
            probability_vector=probability_vector,
            labels=labels,
            caches=caches,
            loss='categorical-cross-entropy',
            activation='sigmoid')

        self.assertEqual(len(grads), 6)
        # Random sample to test reproducibility.
        self.assertAlmostEqual(grads['activation_gradient1'][0][1], 6.91099089)
