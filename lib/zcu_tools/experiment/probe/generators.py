"""Known-model numeric inputs for independent experiment analysis.

All inputs are required. Coordinates and parameters use the same units as the
corresponding fitting model. Seeds initialize a local NumPy generator and never
reseed process-wide state. Returned arrays do not alias caller inputs.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fitting import expfunc


@dataclass(frozen=True)
class DecayParameters:
    """Exponential truth: offset, amplitude in signal units; time in x units."""

    offset: float
    amplitude: float
    decay_time: float


def _coordinates(xs: NDArray[np.float64]) -> NDArray[np.float64]:
    if xs.ndim != 1 or xs.size == 0 or not np.all(np.isfinite(xs)):
        raise ValueError("coordinates must be a nonempty finite 1D array")
    return xs


def _noise(
    shape: tuple[int, ...], noise_std: float, seed: int
) -> NDArray[np.float64]:
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    return np.random.default_rng(seed).normal(0, noise_std, shape)


def generate_decay(
    xs: NDArray[np.float64], truth: DecayParameters, noise_std: float, seed: int
) -> NDArray[np.float64]:
    """Return float64 exponential samples matching the 1D finite xs array.

    truth defines offset/amplitude and positive decay time in xs units.
    noise_std is independent additive Gaussian signal noise per sample; zero
    disables it. seed initializes only this call's RNG. Invalid coordinates,
    nonpositive time or invalid noise raise ValueError. Inputs are unchanged.
    """
    _coordinates(xs)
    if not np.isfinite(truth.decay_time) or truth.decay_time <= 0:
        raise ValueError("decay_time must be finite and positive")
    return expfunc(xs, truth.offset, truth.amplitude, truth.decay_time) + _noise(
        xs.shape, noise_std, seed
    )
