from dataclasses import dataclass

from neural_network.network_functions.math_functions import Array


@dataclass
class ForwardCache:
    """Stores cache from the neural network."""

    def __init__(self, activations: Array, weights: Array, bias: Array) -> None:
        self.activations = activations
        self.weights = weights
        self.bias = bias
