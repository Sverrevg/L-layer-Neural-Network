from unittest import TestCase

import numpy as np

from neural_network.helpers.activation_cache import ActivationCache
from neural_network.helpers.forward_cache import ForwardCache
from neural_network.network_functions.network_functions import l_model_backward, initialize_parameters_deep, \
    l_model_forward, compute_cost, update_parameters


class NetworkFunctionsTestSuite(TestCase):
    def setUp(self) -> None:
        self.input_data = np.array([
            [4.9, 3.0, 1.4, 0.2],
            [-0.2, 1.2, -0.4, 1.4],
            [1.8, -2.3, 5.2, 4.5],
            [6.5, -3.0, 6.3, -0.1]
        ])
        self.probability_vector = np.array([
            [0.1, 0.2, 0.3, .4],
            [0.5, 0.6, 0.9, .3],
            [0.2, 0.1, 0.7, .2],
            [0.7, 0.6, 0.2, .5]
        ])
        self.layer_dims = [4, 3, 3]

    def test_initialize_parameters(self):
        parameters = initialize_parameters_deep(self.layer_dims)

        self.assertEqual(len(parameters), 4)
        self.assertEqual(parameters['Weights1'].size, 12)
        self.assertEqual(parameters['Weights2'].size, 9)
        self.assertEqual(parameters['bias1'].size, 3)
        self.assertEqual(parameters['bias2'].size, 3)

    def test_l_model_forward(self):
        parameters = initialize_parameters_deep(self.layer_dims)
        last_activation_value, caches = l_model_forward(input_data=self.input_data,
                                                        parameters=parameters,
                                                        activation='sigmoid',
                                                        output_shape=3)

        self.assertEqual(last_activation_value.shape, (3, 4))
        for cache in caches:
            self.assertTrue(isinstance(cache, ActivationCache))

        last_activation_value, caches = l_model_forward(input_data=self.input_data,
                                                        parameters=parameters,
                                                        activation='softmax',
                                                        output_shape=3)

        self.assertEqual(last_activation_value.shape, (3, 4))
        for cache in caches:
            self.assertTrue(isinstance(cache, ActivationCache))

    def test_compute_cost(self):
        labels = np.array([[1, 0, 0, 0]])
        cost = compute_cost(probability_vector=self.probability_vector,
                            labels=labels,
                            loss='categorical-cross-entropy')

        self.assertAlmostEqual(first=float(cost),
                               second=1.2405,
                               places=3)

        cost = compute_cost(probability_vector=self.probability_vector,
                            labels=labels,
                            loss='binary-cross-entropy')

        expected = np.array([0.848, 1.067, 0.785, 0.547])
        self.assertAlmostEqual(first=cost.all(),
                               second=expected.all(),
                               places=3)

    def test_l_model_backward(self):
        grads = self.run_l_model_backward(loss='categorical-cross-entropy', activation='sigmoid')

        self.assertEqual(len(grads), 6)
        # Random sample to test reproducibility.
        self.assertAlmostEqual(first=grads['activation_gradient1'][0][1],
                               second=6.285,
                               places=3)

        grads = self.run_l_model_backward(loss='binary-cross-entropy', activation='softmax')

        self.assertEqual(len(grads), 6)
        # Random sample to test reproducibility.
        self.assertAlmostEqual(first=grads['activation_gradient1'][0][1],
                               second=0.586,
                               places=3)

        grads = self.run_l_model_backward(loss='binary-cross-entropy', activation='sigmoid')

        self.assertEqual(len(grads), 6)
        # Random sample to test reproducibility.
        self.assertAlmostEqual(first=grads['activation_gradient1'][0][1],
                               second=6.285,
                               places=3)

    def test_update_parameters(self):
        parameters = initialize_parameters_deep([4, 4, 4])
        grads = self.run_l_model_backward(loss='categorical-cross-entropy', activation='sigmoid')
        new_parameters = update_parameters(parameters=parameters,
                                           grads=grads,
                                           learning_rate=1,
                                           optimizer='stochastic-gradient-descent')

        self.assertEqual(len(new_parameters), 4)

    def run_l_model_backward(self, loss, activation):
        probability_vector = np.array([[0.5, 0.9, 0.4, 0.1]])
        labels = np.array([[1, 0, 0, 0]])
        forward_cache = ForwardCache(activations=self.input_data,
                                     weights=self.input_data,
                                     bias=np.array([[0.1], [-0.5], [0.6], [0.1]]))
        activation_cache = ActivationCache(forward_cache, self.input_data)
        caches = [activation_cache, activation_cache]

        return l_model_backward(
            probability_vector=probability_vector,
            labels=labels,
            caches=caches,
            loss=loss,
            activation=activation)
