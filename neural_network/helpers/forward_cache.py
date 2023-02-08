from dataclasses import dataclass
from typing import Any

import numpy.typing as npt


@dataclass
class ForwardCache:
    """Stores cache from the neural network."""

    def __init__(self, activations: npt.NDArray[Any], weights: npt.NDArray[Any], bias: npt.NDArray[Any]) -> None:
        self.activations = activations
        self.weights = weights
        self.bias = bias
