import numpy as np
import math_operations
import time
from pathlib import Path


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))

        assert (parameters[f'W{l}'].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters[f'b{l}'].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = W.dot(A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    if activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = math_operations.relu(Z)

    elif activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = math_operations.sigmoid(Z)

    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = math_operations.softmax(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters, activation, output_shape):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters[f'W{l}'], parameters[f'b{l}'],
                                             activation="relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID or SOFTMAX. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters[f'W{L}'], parameters[f'b{L}'], activation=activation)
    caches.append(cache)
    assert (AL.shape == (output_shape, X.shape[1]))

    return AL, caches


def compute_cost(AL, Y, loss):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    # Amount of images
    m = Y.shape[1]
    N = AL.shape[1]

    # Cross entropy for binary classification. Different formula:
    if loss == "binary-cross-entropy":
        # Compute loss from aL and y.
        cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

    elif loss == "categorical-cross-entropy":
        # Categorical cross-entropy
        cost = - np.sum(np.multiply(Y, np.log(AL)))
        cost = cost / m

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    # Unpack tuple:
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = math_operations.relu_backward(dA, activation_cache)

    elif activation == "sigmoid":
        dZ = math_operations.sigmoid_backward(dA, activation_cache)

    elif activation == "softmax":
        dZ = math_operations.softmax_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, loss, activation):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    if loss == "binary-cross-entropy":
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID/SOFTMAX -> LINEAR) gradients.
    current_cache = caches[L - 1]
    grads[f'dA{L - 1}'], grads[f'dW{L}'], grads[f'db{L}'] = linear_activation_backward(dAL, current_cache,
                                                                                       activation=activation)

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads[f'dA{l + 1}'], current_cache,
                                                                    activation="relu")
        grads[f'dA{l}'] = dA_prev_temp
        grads[f'dW{l + 1}'] = dW_temp
        grads[f'db{l + 1}'] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters[f'W{l + 1}'] = parameters[f'W{l + 1}'] - learning_rate * grads[f'dW{l + 1}']
        parameters[f'b{l + 1}'] = parameters[f'b{l + 1}'] - learning_rate * grads[f'db{l + 1}']

    return parameters


class NeuralNetwork:
    def __init__(self, layers_dims=[], learning_rate=0.0075, num_iterations=3000, activation="sigmoid",
                 loss="binary-cross-entropy",
                 print_cost=True,
                 save_dir='./../save_files/', filename='parameters.npy'):
        """
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
        save_dir -- default save location. Can be overridden.
        filename -- default file name. Can be overridden.
        """
        self.layers_dims = layers_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost
        self.parameters = []  # Saves trained parameters within the model.
        self.costs = []  # Saves cost within the model after training.
        self.save_dir = save_dir  # Used to load and save the model parameters.
        self.filename = filename
        self.output_activation = activation
        self.loss = loss

        if len(layers_dims) > 0:
            self.output_shape = layers_dims[-1]

    def fit(self, X, Y):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)

        Returns:
        Costs - used to plot costs over time for model insight.
        """
        startTime = time.time()

        np.random.seed(1)
        costs = []  # keep track of cost

        # Parameters initialization.
        parameters = initialize_parameters_deep(self.layers_dims)

        # Loop (gradient descent)
        for i in range(0, self.num_iterations):
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = L_model_forward(X, parameters, self.output_activation, self.output_shape)

            # Compute cost.
            cost = compute_cost(AL, Y, self.loss)

            # Backward propagation.
            grads = L_model_backward(AL, Y, caches, self.loss, self.output_activation)

            # Update parameters.
            self.parameters = update_parameters(parameters, grads, self.learning_rate)

            cost_rounded = np.squeeze(np.round(cost, 3))

            # Print the cost every 100 iterations
            if self.print_cost and i % 100 == 0 or i == self.num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, cost_rounded))
            if i % 100 == 0 or i == self.num_iterations:
                costs.append(np.max(cost))

            # Print cost between 100 iterations:
            print(f'Cost iteration {i}: {cost_rounded}', end='\r')

        execution_time = (time.time() - startTime)
        print('Execution time in seconds: ' + str(round(execution_time, 2)))

        # Save costs to model:
        self.costs = costs

    def test(self, X, y):
        m = X.shape[1]
        p = np.zeros((1, m))

        predictions, caches = L_model_forward(X, self.parameters, self.output_activation, self.output_shape)

        if self.loss == "binary-cross-entropy":
            for i in range(m):
                if predictions[0, i] > 0.5:
                    p[0, i] = 1
                else:
                    p[0, i] = 0

            print("Test accuracy: " + str(np.round(np.sum((p == y) / m))))
        else:
            for i in range(m):
                # Get the index of correct label:
                label = np.where(y[:, i] == 1)[0].item()

                # Get the index of highest probable prediction:
                max = np.where(predictions[:, i] == np.amax(predictions[:, i]))[0].item()

                # Compare. If max does not equal prediction, then assign 0:
                if max == label:
                    p[0, i] = 1
                else:
                    p[0, i] = 0

            predictions_correct = np.sum(p)
            total = np.sum(m)
            accuracy = predictions_correct / total

            print(f'Test accuracy: {accuracy}')

    def predict(self, x):
        AL, caches = L_model_forward(x, self.parameters, self.output_activation, self.output_shape)
        return AL

    def save_model(self):
        print("Saving parameters to", "'" + self.save_dir + self.filename + "'...")

        # Check if save_dir exists, if not, make it:
        Path(self.save_dir).mkdir(exist_ok=True)

        # Numpy.save() saves a numpy array to a file.
        np.save(self.save_dir + self.filename, self.parameters)
        np.save(self.save_dir + "layers_dims.npy", self.layers_dims)

    def load_model(self):
        try:
            # Load saved file into parameters array. Use .item() to retrieve all dictionaries:
            self.parameters = np.load(self.save_dir + self.filename, allow_pickle=True).item()
            self.layers_dims = np.load(self.save_dir + "layers_dims.npy")
            self.output_shape = self.layers_dims[-1]
        except ValueError:
            raise Exception("Parameters cannot be empty.")
