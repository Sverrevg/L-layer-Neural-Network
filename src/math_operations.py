import numpy as np

"""
Source: Coursera DeepLearning course.
"""


def sigmoid(input_data: np.array) -> tuple[[], []]:
    """
    Runs the sigmoid activation on an input (array).

    Arguments:
    input-data -- numpy array of any shape

    Returns:
    output -- output of sigmoid(z), same shape as input data.
    cache -- stores input, useful during backpropagation.
    """
    output = 1 / (1 + np.exp(-input_data))
    cache = input_data

    return output, cache


def relu(input_data):
    """
    Apply ReLU activation on input data.

    Arguments:
    input_data -- array of all values from last layer.

    Returns:
    output -- Post-activation parameter, of the same shape as the input data.
    cache -- a python dictionary containing the input data; stored for computing the backward pass efficiently.
    """
    output = np.maximum(0, input_data)

    assert output.shape == input_data.shape

    cache = input_data
    return output, cache


def softmax(input_data):
    """"
    Apply softmax activation to input data, for multiclass classification.

    Arguments:
    input_data -- array of all values from last layer.
    
    Returns:
    output -- Softmax values of the input data.
    cache -- the input data; stored for computing the backward pass efficiently.
    """
    shift_z = input_data - np.max(input_data)  # Shifted value from last layer.
    exponent = np.exp(shift_z)  # Calculate exponent of z value.
    output = exponent / np.sum(exponent)
    cache = input_data

    return output, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def softmax_backward(dA, cache):
    """
    Implement the backward propagation for a softmax layer.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z

    Source: https://e2eml.school/softmax.html
    """
    Z = cache

    s = np.exp(Z) / (np.sum(np.exp(Z)))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ
