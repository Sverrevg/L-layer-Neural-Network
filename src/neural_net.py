import numpy as np

import math_operations

"""
Based on the Neural Networks and Deep Learning courses on Coursera.
"""


def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(5)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

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
        cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

    Z = np.dot(W, A) + b
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
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A, activation_cache = math_operations.sigmoid(Z)

        # YOUR CODE ENDS HERE

    elif activation == "relu":
        A, activation_cache = math_operations.relu(Z)

        # YOUR CODE ENDS HERE
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    # The for loop starts at 1 because layer 0 is the input
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,
                                             parameters[f'W{l}'],
                                             parameters[f'b{l}'],
                                             activation="relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A,
                                          parameters[f'W{L}'],
                                          parameters[f'b{L}'],
                                          activation="sigmoid")
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = -(1 / m) * (np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL))))

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

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

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
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = math_operations.relu_backward(dA, activation_cache)

    elif activation == "sigmoid":
        dZ = math_operations.sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


class neural_net:
    def __init__(self, input_dim, layers):
        self.layer_count = len(layers)
        self.W_array = []
        self.b_array = []

    # def forward_propagation(self, X, parameters):
    #     """
    #     Argument:
    #     X -- input data of size (n_x, m)
    #     parameters -- python dictionary containing your parameters (output of initialization function)
    #
    #     Returns:
    #     S -- The sigmoid output
    #     cache -- a dictionary containing Z[i] and A[i]
    #
    #     """
    #
    #     # Retrieve all parameters and store in array. Each index represents one layer:
    #     W_array = []
    #     b_array = []
    #
    #     for i in range(self.layer_count):
    #         W_array.append(parameters[f"W{i}"])
    #         b_array.append(parameters[f"b{i}"])
    #
    #     # Store all Z and A values:
    #     Z_array = []
    #     cache_array = []
    #
    #     # Calculate A for each layer:
    #     for i in range(self.layer_count):
    #         Z = np.dot(W_array[i], X) + b_array[i]
    #         A = np.tanh(Z)  # tanh activation function
    #         Z_array.append(Z)
    #         cache = {f"Z{i}": Z,
    #                  f"A{i}": A}
    #         cache_array.append(cache)
    #
    #     # Calculate S using last Z element:
    #     S = math_operations.sigmoid(Z_array[-1])
    #
    #     return S, cache_array
    #
    # def compute_cost(self, S, Y):
    #     """
    #     Computes the cross-entropy cost
    #
    #     Arguments:
    #     S -- The sigmoid output of the second activation, of shape (1, number of examples)
    #     Y -- "true" labels vector of shape (1, number of examples)
    #
    #     Returns:
    #     cost -- cross-entropy cost given equation (13)
    #
    #     """
    #     m = Y.shape[1]  # number of examples
    #
    #     # Compute the cross-entropy cost
    #     logprobs = np.multiply(np.log(S), Y) + np.multiply((1 - Y), np.log(1 - S))
    #     cost = -(1 / m) * np.sum(logprobs)
    #
    #     cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. E.g., turns [[17]] into 17
    #
    #     return cost
    #
    # def backward_propagation(self, parameters, cache, X, Y):
    #     """
    #     Implement the backward propagation using the instructions above.
    #
    #     Arguments:
    #     parameters -- python dictionary containing our parameters
    #     cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    #     X -- input data of shape (2, number of examples)
    #     Y -- "true" labels vector of shape (1, number of examples)
    #
    #     Returns:
    #     grads -- python dictionary containing your gradients with respect to different parameters
    #     """
    #     m = X.shape[1]
    #
    #     # Retrieve all parameters and store in array. Each index represents one layer:
    #     W_array = []
    #
    #     for i in range(self.layer_count):
    #         W_array.append(parameters[f"W{i}"])
    #
    #     # Store all Z and A values:
    #     A_array = []
    #
    #     # Calculate A for each layer:
    #     for i in range(self.layer_count):
    #         A_array.append(cache[f"A{i}"])
    #
    #     # Store dZ, dW and dB values:
    #     dZ_array = []
    #     dW_array = []
    #     dB_array = []
    #
    #     # Calculate dZN first. This is from the output layer:
    #     dZ = A_array[-1] - Y
    #     size = len(self.layer_count)
    #
    #     for i in range(self.layer_count):
    #         dW = (1 / m) * np.dot(dZ, A_array[i].T)
    #         dB = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    #         dZ = np.dot(W_array[size - i])
    #
    # def update_parameters(parameters, grads, learning_rate=1.2):
    #     """
    #     Updates parameters using the gradient descent update rule given above
    #
    #     Arguments:
    #     parameters -- python dictionary containing your parameters
    #     grads -- python dictionary containing your gradients
    #
    #     Returns:
    #     parameters -- python dictionary containing your updated parameters
    #
    #     """
    #     return 0
    #
    # def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    #     """
    #     Arguments:
    #     X -- dataset of shape (2, number of examples)
    #     Y -- labels of shape (1, number of examples)
    #     n_h -- size of the hidden layer
    #     num_iterations -- Number of iterations in gradient descent loop
    #     print_cost -- if True, print the cost every 1000 iterations
    #
    #     Returns:
    #     parameters -- parameters learnt by the model. They can then be used to predict.
    #
    #     """
    #     return 0
