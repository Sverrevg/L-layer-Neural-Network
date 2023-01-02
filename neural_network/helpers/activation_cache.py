from dataclasses import dataclass

from neural_network.helpers.forward_cache import ForwardCache
from neural_network.math_operations import ndarray


@dataclass
class ActivationCache:
    """Stores cache from the neural network."""

    def __init__(self, linear_cache: ForwardCache, activation_cache: ndarray) -> None:
        self.linear_cache = linear_cache
        self.activation_cache = activation_cache
