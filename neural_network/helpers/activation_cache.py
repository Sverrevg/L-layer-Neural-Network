from dataclasses import dataclass
from typing import Any

import numpy.typing as npt

from neural_network.helpers.forward_cache import ForwardCache


@dataclass
class ActivationCache:
    """Stores cache from the neural network."""

    def __init__(self, linear_cache: ForwardCache, activation_cache: npt.NDArray[Any]) -> None:
        self.linear_cache = linear_cache
        self.activation_cache = activation_cache
