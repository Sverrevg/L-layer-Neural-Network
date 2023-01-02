import time
from pathlib import Path

import numpy as np

from neural_network.network_operations.activation import Activation
from neural_network.network_operations.loss import Loss
from neural_network.network_operations.network_operations import (
    compute_cost,
    initialize_parameters_deep,
    l_model_backward,
    l_model_forward,
    update_parameters,
)
from neural_network.network_operations.optimizer import Optimizer


class NeuralNetwork:
    def __init__(self,
                 layers_dims=None,
                 learning_rate=0.0075,
                 num_iterations=3000,
                 activation=Activation.SIGMOID.value,
                 loss=Loss.BINARY.value,
                 optimizer=Optimizer.SGDM,
                 print_cost=True,
                 beta=0.5,
                 save_dir='./../save_files/',
                 filename='parameters.npy'):
        """
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
        save_dir -- default save location. Can be overridden.
        filename -- default file name. Can be overridden.
        """

        if layers_dims is None:
            layers_dims = []
        self.layers_dims = layers_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        self.parameters = []  # Saves trained parameters within the model.
        self.momentum = {}  # Saves momenta within the model.
        self.costs = []  # Saves cost within the model after training.
        self.save_dir = save_dir  # Used to load and save the model parameters.
        self.filename = filename
        self.output_activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.beta = beta

        if len(layers_dims) > 0:
            self.output_shape = layers_dims[-1]

    def fit(self, input_data, labels):
        """
        Implements an L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        input_data -- data, numpy array of shape (num_px * num_px * 3, number of examples)
        labels -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)

        Returns:
        Costs - used to plot costs over time for model insight.
        """
        start_time = time.time()

        np.random.seed(1)
        costs = []  # keep track of cost

        # Parameters initialization.
        parameters = initialize_parameters_deep(self.layers_dims)

        # Loop (gradient descent)
        for i in range(0, self.num_iterations):
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            probability_vector, caches = l_model_forward(input_data, parameters, self.output_activation,
                                                         self.output_shape)

            # Compute cost.
            cost = compute_cost(probability_vector, labels, self.loss)

            # Backward propagation.
            grads = l_model_backward(probability_vector, labels, caches, self.loss, self.output_activation)

            # Update parameters.
            self.parameters, self.momentum = update_parameters(parameters, grads, self.momentum, self.learning_rate,
                                                               self.optimizer, i, self.beta)

            cost_rounded = np.squeeze(np.round(cost, 3))

            # Print the cost every 100 iterations
            if self.print_cost and i % 100 == 0 or i == self.num_iterations - 1:
                print(f"Cost after iteration {i}: {cost_rounded}")
            if i % 100 == 0 or i == self.num_iterations:
                costs.append(np.max(cost))

            # Print cost between 100 iterations:
            print(f'Cost iteration {i}: {cost_rounded}', end='\r')

        execution_time = (time.time() - start_time)
        print('Execution time in seconds: ' + str(round(execution_time, 2)))

        # Save costs to model:
        self.costs = costs

    def test(self, input_data, labels):
        input_shape = input_data.shape[1]
        predictions = np.zeros((1, input_shape))

        outputs, _ = l_model_forward(input_data, self.parameters, self.output_activation, self.output_shape)

        if self.loss == Loss.BINARY.value:
            for i in range(input_shape):
                if outputs[0, i] > 0.5:
                    predictions[0, i] = 1
                else:
                    predictions[0, i] = 0

            print("Test accuracy: " + str(np.round(np.sum((predictions == labels) / input_shape))))
        else:
            for i in range(input_shape):
                # Get the index of correct label:
                label = np.where(labels[:, i] == 1)[0].item()

                # Get the index of the highest probable prediction:
                prediction = np.where(outputs[:, i] == np.amax(outputs[:, i]))[0].item()

                # Compare. If max does not equal prediction, then assign 0:
                if prediction == label:
                    predictions[0, i] = 1
                else:
                    predictions[0, i] = 0

            predictions_correct = np.sum(predictions)
            total = np.sum(input_shape)
            accuracy = predictions_correct / total

            print(f'Test accuracy: {accuracy}')

    def predict(self, input_data):
        probability_vector, _ = l_model_forward(input_data, self.parameters, self.output_activation, self.output_shape)
        return probability_vector

    def save_model(self) -> None:
        """
        Save the trained weights and biases to files that can be loaded later.
        """
        print("Saving parameters to", "'" + self.save_dir + self.filename + "'...")

        # Check if save_dir exists, if not, make it:
        Path(self.save_dir).mkdir(exist_ok=True)

        # Numpy.save() saves a numpy array to a file.
        np.save(self.save_dir + self.filename, self.parameters)
        np.save(self.save_dir + "layers_dims.npy", self.layers_dims)

    def load_model(self) -> None:
        """
        Load trained weights and biases into existing model.
        """
        try:
            # Load saved file into parameters array. Use .item() to retrieve all dictionaries:
            self.parameters = np.load(self.save_dir + self.filename, allow_pickle=True).item()
            self.layers_dims = np.load(self.save_dir + "layers_dims.npy")
            self.output_shape = self.layers_dims[-1]
        except ValueError as exc:
            raise ValueError("Parameters cannot be empty.") from exc
