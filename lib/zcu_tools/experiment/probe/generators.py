"""Known-model numeric inputs for independent experiment analysis.

All inputs are required. Coordinates and parameters use the same units as the
corresponding fitting model. Seeds initialize a local NumPy generator and never
reseed process-wide state. Returned arrays do not alias caller inputs.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fitting import cosfunc, decaycos, expfunc, lorfunc


@dataclass(frozen=True)
class DecayParameters:
    """Exponential truth: offset, amplitude in signal units; time in x units."""

    offset: float
    amplitude: float
    decay_time: float


@dataclass(frozen=True)
class OscillationParameters:
    """Cosine truth in fitting coordinates.

    offset/amplitude are in signal units; frequency is cycles per x unit and
    phase_degrees is the phase at x=0. decay_time is positive in x units, or
    None for constant envelope (supported by generate_rabi only).
    """

    offset: float
    amplitude: float
    frequency: float
    phase_degrees: float
    decay_time: float | None


@dataclass(frozen=True)
class LorentzianParameters:
    """Lorentzian truth: offset/amplitude in signal units, slope in signal/x.

    center and positive half_width use x units; offset is the baseline at center.
    """

    offset: float
    slope: float
    amplitude: float
    center: float
    half_width: float


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


def generate_decay_fringe(
    xs: NDArray[np.float64], truth: OscillationParameters, noise_std: float, seed: int
) -> NDArray[np.float64]:
    """Return a float64 damped cosine matching finite 1D xs.

    truth uses the fitting decaycos convention; decay_time must be positive,
    not None. noise_std is additive independent Gaussian signal noise and seed
    initializes a local RNG. Invalid coordinates, time or noise raise ValueError.
    No inputs are modified.
    """
    _coordinates(xs)
    if (
        truth.decay_time is None
        or not np.isfinite(truth.decay_time)
        or truth.decay_time <= 0
    ):
        raise ValueError("decay_time must be finite and positive")
    return decaycos(
        xs,
        truth.offset,
        truth.amplitude,
        truth.frequency,
        truth.phase_degrees,
        truth.decay_time,
    ) + _noise(xs.shape, noise_std, seed)


def generate_rabi(
    xs: NDArray[np.float64], truth: OscillationParameters, noise_std: float, seed: int
) -> NDArray[np.float64]:
    """Return float64 Rabi samples matching finite 1D xs without changing inputs.

    None decay_time selects fitting.cosfunc, otherwise fitting.decaycos with
    positive time. Frequency is cycles per x unit, phase is in degrees.
    noise_std is independent additive Gaussian signal noise per sample; seed
    initializes a local RNG. Invalid coordinates, time or noise raise ValueError.
    """
    if truth.decay_time is not None:
        return generate_decay_fringe(xs, truth, noise_std, seed)
    _coordinates(xs)
    return cosfunc(
        xs, truth.offset, truth.amplitude, truth.frequency, truth.phase_degrees
    ) + _noise(xs.shape, noise_std, seed)


def generate_lorentzian(
    xs: NDArray[np.float64], truth: LorentzianParameters, noise_std: float, seed: int
) -> NDArray[np.float64]:
    """Return fitting.lorfunc samples on finite 1D xs, without changing inputs.

    truth supplies a peak (positive amplitude) or dip (negative amplitude) with
    linear background and positive half_width in xs units. noise_std is the
    independent additive Gaussian signal noise; seed initializes a local RNG.
    Invalid coordinates, width or noise raise ValueError.
    """
    _coordinates(xs)
    if not np.isfinite(truth.half_width) or truth.half_width <= 0:
        raise ValueError("half_width must be finite and positive")
    return lorfunc(
        xs, truth.offset, truth.slope, truth.amplitude, truth.center, truth.half_width
    ) + _noise(xs.shape, noise_std, seed)
