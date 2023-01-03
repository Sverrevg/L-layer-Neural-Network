import time
from pathlib import Path

import numpy as np

from neural_network.math_functions import Array
from neural_network.network_operations.activation import Activation
from neural_network.network_operations.loss import Loss
from neural_network.network_operations.network_functions import (
    compute_cost,
    initialize_parameters_deep,
    l_model_backward,
    l_model_forward,
    update_parameters,
)
from neural_network.network_operations.optimizer import Optimizer


class NeuralNetwork:
    def __init__(self,
                 layers_dims: list[int],
                 learning_rate: float = 0.0075,
                 num_iterations: int = 3000,
                 activation: str = Activation.SIGMOID.value,
                 loss: str = Loss.BINARY.value,
                 optimizer: str = Optimizer.SGDM.value,
                 print_cost: bool = True,
                 beta: float = 0.5,
                 save_dir: str = './../save_files/',
                 parameters_filename: str = 'parameters.npy',
                 dims_filename: str = 'layers_dims.npy'):
        """
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
        save_dir -- default save location. Can be overridden.
        filename -- default file name. Can be overridden.
        """
        # Input parameters:
        self.layers_dims: list[int] = layers_dims
        self.learning_rate: float = learning_rate
        self.num_iterations: int = num_iterations
        self.activation: str = activation
        self.loss: str = loss
        self.optimizer: str = optimizer
        self.print_cost: bool = print_cost
        self.beta: float = beta
        self.save_dir: str = save_dir  # Used to load and save the model parameters.
        self.parameters_filename: str = parameters_filename
        self.dims_filename: str = dims_filename

        # Network stores:
        self.parameters: dict[str, Array] = {}  # Saves trained parameters within the model.
        self.momentum: dict[str, Array] = {}  # Saves momenta within the model.
        self.costs: list[float] = []  # Saves cost within the model after training.

        if len(layers_dims) > 0:
            self.output_shape = layers_dims[-1]

    def fit(self, input_data: Array, labels: Array) -> None:
        """
        Implements an L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        input_data -- data, numpy array of shape (num_px * num_px * 3, number of examples)
        labels -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)

        Returns:
        Costs - used to plot costs over time for model insight.
        """
        start_time = time.time()

        costs: list[float] = []  # keep track of cost

        # Parameters initialization.
        parameters = initialize_parameters_deep(self.layers_dims)

        # Loop (gradient descent)
        for i in range(0, self.num_iterations):
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            probability_vector, caches = l_model_forward(input_data, parameters, self.activation,
                                                         self.output_shape)

            # Compute cost.
            cost = compute_cost(probability_vector, labels, self.loss)

            # Backward propagation.
            grads = l_model_backward(probability_vector, labels, caches, self.loss, self.activation)

            # Update parameters.
            self.parameters, self.momentum = update_parameters(parameters, grads, self.momentum, self.learning_rate,
                                                               self.optimizer, i, self.beta)

            cost_rounded = np.squeeze(np.round(cost, 3))

            # Print the cost every 100 iterations
            if self.print_cost and i % 100 == 0 or i == self.num_iterations - 1:
                print(f"Cost after iteration {i}: {cost_rounded}")
            if i % 100 == 0 or i == self.num_iterations:
                costs.append(float(np.max(cost)))

            # Print cost between 100 iterations:
            print(f'Cost iteration {i}: {cost_rounded}', end='\r')

        execution_time = (time.time() - start_time)
        print('Execution time in seconds: ' + str(round(execution_time, 2)))

        # Save costs to model:
        self.costs = costs

    def test(self, input_data: Array, labels: Array) -> None:
        input_shape = input_data.shape[1]
        predictions = np.zeros((1, input_shape))

        outputs, _ = l_model_forward(input_data, self.parameters, self.activation, self.output_shape)

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

    def predict(self, input_data: Array) -> Array:
        probability_vector, _ = l_model_forward(input_data, self.parameters, self.activation, self.output_shape)
        return probability_vector

    def save_model(self) -> None:
        """
        Save the trained weights and biases to files that can be loaded later.
        """
        print("Saving parameters to", "'" + self.save_dir + self.parameters_filename + "'...")

        # Check if save_dir exists, if not, make it:
        Path(self.save_dir).mkdir(exist_ok=True)

        # Numpy.save() saves a numpy array to a file.
        np.save(self.save_dir + self.parameters_filename, self.parameters)  # type: ignore
        np.save(self.save_dir + self.dims_filename, self.layers_dims)

    def load_model(self, parameters_filename: str, dims_filename: str) -> None:
        """
        Load trained weights and biases into existing model.
        """
        try:
            # Load saved file into parameters array. Use .item() to retrieve all dictionaries:
            self.parameters = np.load(self.save_dir + parameters_filename, allow_pickle=True).item()
            self.layers_dims = np.load(self.save_dir + dims_filename)
            self.output_shape = self.layers_dims[-1]
        except ValueError as exc:
            raise ValueError("Parameters cannot be empty.") from exc
