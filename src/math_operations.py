import numpy as np

"""
Source: Coursera DeepLearning course.
"""


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def softmax(Z):
    """"
    Implement softmax function for multiclass classification.

    Arguments:
    z -- single value from last layer
    Z -- array of all values from last layer
    """
    """Compute the softmax of vector x in a numerically stable way."""
    shiftZ = Z - np.max(Z)
    A = np.exp(shiftZ)
    cache = Z

    return A / np.sum(A), cache


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
    Z_softmax = np.exp(Z) / np.sum(np.exp(Z))
    s_reshaped = np.reshape(Z_softmax, (1, -1))
    grad = np.reshape(dA, (1, -1))

    # The @ symbol calls NumPy's matrix multiplication function.
    # The matrix product of the N x 1 softmax transpose and the 1 x N softmax is an N x N two dimensional array.
    d_softmax = (s_reshaped * np.identity(s_reshaped.size) - s_reshaped.transpose() @ s_reshaped)

    # Thanks to this careful setup, we can now calculate the input gradient with just one more matrix multiplication.
    dZ = grad @ d_softmax

    return dZ
