from dataclasses import dataclass

from neural_network.math_operations import ndarray


@dataclass
class ForwardCache:
    """Stores cache from the neural network."""

    def __init__(self, activations: ndarray, weights: ndarray, bias: ndarray) -> None:
        self.activations = activations
        self.weights = weights
        self.bias = bias
