import numpy as np


class NeuralNetwork:
    def __init__(self, learning_rate):
        # Function np.random.randn(): Return a sample (or samples) from the “standard normal” distribution.
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for iteration in range(iterations):
            # Pick a random data instance.
            random_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_index]
            target = targets[random_index]

            # Compute the gradients and update the weights.
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances.
            # Check if iteration is multiple of 100.
            if iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all instances to measure the error.
                for index in range(len(input_vectors)):
                    data_point = input_vectors[index]
                    target = targets[index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors

    # Function np.exp: Calculate the exponential of all elements in the input array.
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
                derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
                derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
                derror_dweights * self.learning_rate
        )
