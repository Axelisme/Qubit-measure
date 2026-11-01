"""Caller-model synthesis and population-based raw IQ sampling."""

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


def synthesize(
    model: Callable[..., NDArray[np.number]],
    xs: NDArray[np.number],
    params: Sequence[float],
    *,
    noise_std: float,
    seed: int,
) -> NDArray[np.number]:
    """Call model(xs, *params) to produce an array with the model's shape."""
    return model(xs, *params)
