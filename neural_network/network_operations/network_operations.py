import numpy as np

from neural_network import math_operations
from neural_network.helpers.activation_cache import ActivationCache
from neural_network.helpers.forward_cache import ForwardCache
from neural_network.math_operations import ndarray
from neural_network.network_operations.activation import Activation
from neural_network.network_operations.loss import Loss
from neural_network.network_operations.optimizer import Optimizer


def initialize_parameters_deep(layer_dims: list[int]) -> dict[str, ndarray]:
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network.

    Returns:
    parameters -- python dictionary containing network parameters (weights and bias):
        * Weights{layer} -- weight matrix of shape (layer_dims[l], layer_dims[l-1]).
        * bias{layer} -- bias vector of shape (layer_dims[l], 1).
    """
    np.random.seed(1)
    parameters = {}
    layer_count = len(layer_dims)  # Number of layers in the network.

    for layer in range(1, layer_count):
        parameters[f'Weights{layer}'] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) / np.sqrt(
            layer_dims[layer - 1])
        parameters[f'bias{layer}'] = np.zeros((layer_dims[layer], 1))

        assert parameters[f'Weights{layer}'].shape == (layer_dims[layer], layer_dims[layer - 1])
        assert parameters[f'bias{layer}'].shape == (layer_dims[layer], 1)

    return parameters


def linear_forward(activations: ndarray, weights: ndarray, bias: ndarray) -> \
        tuple[ndarray, ForwardCache]:
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    activations -- activations from previous layer (or input data): (size of previous layer, number of examples).
    weights -- weights matrix: numpy array of shape (size of current layer, size of previous layer).
    bias -- bias vector, numpy array of shape (size of the current layer, 1).

    Returns:
    activation_input -- the input of the activation function, also called pre-activation parameter.
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently.
    """
    activation_input = weights.dot(activations) + bias
    cache = ForwardCache(activations, weights, bias)

    return activation_input, cache


def linear_activation_forward(activations_prev: ndarray, weights: ndarray, bias: ndarray,
                              activation: str) -> tuple[ndarray, ActivationCache]:
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    activations_prev -- activations from previous layer (or input data): (size of previous layer, number of examples).
    weights -- weights matrix: numpy array of shape (size of current layer, size of previous layer).
    bias -- bias vector, numpy array of shape (size of the current layer, 1).
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu".

    Returns:
    outputs -- the output of the activation function, also called the post-activation value.
    cache -- a python dictionary containing "linear_cache" and "activation_cache"; stored for computing the backward
    pass efficiently.
    """
    if activation == Activation.RELU.value:
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        inputs, linear_cache = linear_forward(activations_prev, weights, bias)
        outputs, activation_cache = math_operations.relu(inputs)

    elif activation == Activation.SIGMOID.value:
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        inputs, linear_cache = linear_forward(activations_prev, weights, bias)
        outputs, activation_cache = math_operations.sigmoid(inputs)

    elif activation == Activation.SOFTMAX.value:
        inputs, linear_cache = linear_forward(activations_prev, weights, bias)
        outputs, activation_cache = math_operations.softmax(inputs)

    return outputs, ActivationCache(linear_cache, activation_cache)


def l_model_forward(input_data: ndarray, parameters: dict[str, ndarray], activation: str, output_shape: int) -> \
        tuple[ndarray, list[ActivationCache]]:
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation.

    Arguments:
    input_data -- data, numpy array of shape (input size, number of examples).
    parameters -- output of initialize_parameters_deep().
    activation -- type of activation function to use.
    output_shape -- shape of the output data.

    Returns:
    last_activation_value -- last post-activation value.
    caches -- list of caches containing:
        * every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2).
        * the cache of linear_sigmoid_forward() (there is one, indexed L-1).
    """

    caches = []
    layer_count = len(parameters) // 2

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for layer in range(1, layer_count):
        a_prev = input_data
        input_data, cache = linear_activation_forward(a_prev, parameters[f'Weights{layer}'], parameters[f'bias{layer}'],
                                                      activation="relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID or SOFTMAX. Add "cache" to the "caches" list.
    last_activation_value, cache = linear_activation_forward(input_data, parameters[f'Weights{layer_count}'],
                                                             parameters[f'bias{layer_count}'], activation=activation)
    caches.append(cache)
    assert last_activation_value.shape == (output_shape, input_data.shape[1])

    return last_activation_value, caches


def compute_cost(probability_vector: ndarray, label: ndarray, loss: str) -> ndarray:
    """
    Implement the cost function defined by equation (7).

    Arguments:
    probability_vector -- probability vector corresponding to your label predictions, shape (1, number of examples).
    label -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples).

    Returns:
    cost -- cross-entropy cost.
    """
    # Amount of images
    input_shape = label.shape[1]
    cost = 0.

    # Cross entropy for binary classification. Different formula:
    if loss == Loss.BINARY.value:
        # Compute loss from aL and y.
        cost = (1. / input_shape) * (
                -np.dot(label, np.log(probability_vector).T) - np.dot(1 - label, np.log(1 - probability_vector).T))

    elif loss == Loss.CATEGORICAL.value:
        # Categorical cross-entropy
        cost = - np.sum(np.multiply(label, np.log(probability_vector)))
        cost = cost / input_shape

    return np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).


def linear_backward(cost_gradient: ndarray, cache: ForwardCache) -> tuple[ndarray, ndarray, ndarray]:
    """
    Implement the linear portion of backward propagation for a single layer (layer l).

    Arguments:
    cost_gradient -- Gradient of the cost with respect to the linear output (of current layer l).
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer.

    Returns:
    weight_gradient -- Gradient of the cost with respect to W (current layer l), same shape as W.
    bias_gradient -- Gradient of the cost with respect to b (current layer l), same shape as b.
    activation_gradient -- Gradient of the cost with respect to the activation (of the previous layer l-1),
    same shape as activations_prev.
    """
    input_shape = cache.activations.shape[1]

    weight_gradient = 1. / input_shape * np.dot(cost_gradient, cache.activations.T)
    bias_gradient = 1. / input_shape * np.sum(cost_gradient, axis=1, keepdims=True)
    activation_gradient = np.dot(cache.weights.T, cost_gradient)

    assert activation_gradient.shape == cache.activations.shape
    assert weight_gradient.shape == cache.weights.shape
    assert bias_gradient.shape == cache.bias.shape

    return activation_gradient, weight_gradient, bias_gradient


def linear_activation_backward(post_activation_gradient: ndarray, cache: ActivationCache, activation: str) -> \
        tuple[ndarray, ndarray, ndarray]:
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    post_activation_gradient -- post-activation gradient for current layer l.
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently.
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu".

    Returns:
    activation_gradient -- Gradient of the cost with respect to the activation (of the previous layer l-1),
    same shape as A_prev.
    weight_gradient -- Gradient of the cost with respect to W (current layer l), same shape as W.
    bias_gradient -- Gradient of the cost with respect to b (current layer l), same shape as b.
    """
    cost_gradient = np.array(0)

    if activation == Activation.RELU.value:
        cost_gradient = math_operations.relu_backward(post_activation_gradient, cache.activation_cache)

    elif activation == Activation.SIGMOID.value:
        cost_gradient = math_operations.sigmoid_backward(post_activation_gradient, cache.activation_cache)

    elif activation == Activation.SOFTMAX.value:
        cost_gradient = math_operations.softmax_backward(post_activation_gradient, cache.activation_cache)

    activation_gradient, weight_gradient, bias_gradient = linear_backward(cost_gradient, cache.linear_cache)

    return activation_gradient, weight_gradient, bias_gradient


def l_model_backward(probability_vector: ndarray, label: ndarray, caches: list[ActivationCache], loss: str,
                     activation: str) -> dict[str, ndarray]:
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group.

    Arguments:
    probability_vector -- probability vector, output of the forward propagation (L_model_forward()).
    label -- true "label" vector (containing 0 if non-cat, 1 if cat).
    caches -- list of caches containing:
        * every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2).
        * the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1).

    Returns:
    grads -- A dictionary with the gradients:
        * grads["activation_gradient" + str(l)] = ...
        * grads["weight_gradient" + str(l)] = ...
        * grads["bias_gradient" + str(l)] = ...
    """
    grads = {}
    layer_count = len(caches)  # The total number of layers.
    # probability_vector.shape[1]
    if loss == Loss.BINARY.value:
        label = label.reshape(probability_vector.shape)  # After this line, Y is the same shape as AL.

    # Initializing the backpropagation:
    post_activation_gradient = - (np.divide(label, probability_vector) - np.divide(1 - label, 1 - probability_vector))

    # Nth layer (SIGMOID/SOFTMAX -> LINEAR) gradients.
    current_cache = caches[layer_count - 1]
    grads[f'activation_gradient{layer_count - 1}'], grads[f'weight_gradient{layer_count}'], grads[
        f'bias_gradient{layer_count}'] = linear_activation_backward(post_activation_gradient, current_cache,
                                                                    activation=activation)

    for layer in reversed(range(layer_count - 1)):
        # Nth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[layer]
        activation_gradient_temp, weight_gradient_temp, bias_gradient_temp = linear_activation_backward(
            grads[f'activation_gradient{layer + 1}'], current_cache, activation="relu")
        grads[f'activation_gradient{layer}'] = activation_gradient_temp
        grads[f'weight_gradient{layer + 1}'] = weight_gradient_temp
        grads[f'bias_gradient{layer + 1}'] = bias_gradient_temp

    return grads


def update_parameters(parameters: dict[str, ndarray], grads: dict[str, ndarray], momentum: dict[str, ndarray],
                      learning_rate: float, optimizer: str, iteration: int, beta: float) -> \
        tuple[dict[str, ndarray], dict[str, ndarray]]:
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters.
    grads -- python dictionary containing your gradients, output of L_model_backward.
    momentum -- python dictionary containing momenta from previous iteration.

    Returns:
    parameters -- python dictionary containing your updated parameters:
        * parameters["Weight" + str(l)] = ...
        * parameters["bias" + str(l)] = ...
    """

    layer_count = len(parameters) // 2

    if optimizer == Optimizer.SGD.value:
        # Update rule for each parameter. Use a for loop.
        for layer in range(layer_count):
            parameters[f'Weights{layer + 1}'] = parameters[f'Weights{layer + 1}'] - learning_rate * grads[
                f'weight_gradient{layer + 1}']
            parameters[f'bias{layer + 1}'] = parameters[f'bias{layer + 1}'] - learning_rate * grads[
                f'bias_gradient{layer + 1}']

    # Calculate v:
    if optimizer == Optimizer.SGDM.value:
        # At first iteration vw and vb equal gradients for each layer:
        for layer in range(layer_count):
            layer += 1

            if iteration == 0:
                # Should only save one instance:
                momentum[f'vw{layer}'] = grads[f'weight_gradient{layer}']
                momentum[f'vb{layer}'] = grads[f'bias_gradient{layer}']

            elif iteration > 0:
                # Calculate new momentum with values from previous iteration:
                momentum[f'vw{layer}'] = beta * momentum[f'vw{layer}'] + grads[f'weight_gradient{layer}']
                momentum[f'vb{layer}'] = beta * momentum[f'vb{layer}'] + grads[f'bias_gradient{layer}']

            # Update each parameter:
            parameters[f'Weights{layer}'] = parameters[f'Weights{layer}'] - learning_rate * momentum[f'vw{layer}']
            parameters[f'bias{layer}'] = parameters[f'bias{layer}'] - learning_rate * momentum[f'vb{layer}']

    return parameters, momentum
