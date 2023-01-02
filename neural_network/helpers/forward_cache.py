from dataclasses import dataclass
from typing import Any

from neural_network.helpers.array import Array


@dataclass
class ForwardCache:
    """Stores cache from the neural network."""

    def __init__(self, activations: Array[Any, float], weights: Array[Any, float], bias) -> None:
        self.activations = activations
        self.weights = weights
        self.bias = bias
