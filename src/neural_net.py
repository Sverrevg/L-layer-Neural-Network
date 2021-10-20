import numpy as np

import math_operations


def initialize_parameters(n_x, layers):
    """
    Argument:
    n_x -- size of the input layer
    layers -- sizes of the other layers

    Returns:
    params -- python dictionary containing your parameters

    """
    np.random.seed(2)
    parameters = {}

    # Init for all layers:
    for i, s in enumerate(layers):
        W = np.random.randn(s, n_x) * 0.01
        b = np.zeros((s, 1))

        # Add parameters to dict:
        parameters.update({f"W{i}": W,
                           f"b{i}": b})

    return parameters


def forward_propagation(X, parameters, layer_count):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    S -- The sigmoid output
    cache -- a dictionary containing Z[i] and A[i]

    """

    # Retrieve all parameters and store in array. Each index contains one layer:
    W_array = []
    b_array = []

    for i in range(layer_count):
        W_array.append(parameters[f"W{i}"])
        b_array.append(parameters[f"b{i}"])

    # Store all Z and A values:
    Z_array = []
    A_array = []

    # Calculate A:
    for i in range(layer_count):
        Z = np.dot(W_array[i], X) + b_array[i]
        A = np.tanh(Z)  # tanh activation function
        Z_array.append(Z)
        A_array.append(A)

    # Calculate S using last Z element:
    S = math_operations.sigmoid(Z[-1])

    return S, Z_array, A_array


def compute_cost(S, Y):
    """
    Computes the cross-entropy cost

    Arguments:
    S -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost given equation (13)

    """
    m = Y.shape[1]  # number of examples

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(S), Y) + np.multiply((1 - Y), np.log(1 - S))
    cost = -(1 / m) * np.sum(logprobs)

    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. E.g., turns [[17]] into 17

    return cost
