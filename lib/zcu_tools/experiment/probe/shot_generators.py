"""Raw-IQ sampling from the public singleshot fitting distributions."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fitting import gauss_func
from zcu_tools.analysis.fitting.singleshot_rabi import (
    RabiPhysicalParams,
    model_bin_probabilities,
)


@dataclass(frozen=True)
class GaussianCloudParameters:
    """Two circular Gaussian IQ clouds with symmetric preparation errors.

    center_g/center_e are complex IQ centers. width is positive standard
    deviation in each quadrature, in the same IQ units. shots is the positive
    number of samples per prepared state. misassignment in [0,1] is the
    probability of drawing from the opposite cloud in either prepared row.
    All fields are required.
    """

    center_g: complex
    center_e: complex
    width: float
    shots: int
    misassignment: float


def _iq_axis(axis: NDArray[np.float64]) -> None:
    if (
        axis.ndim != 1 or axis.size < 3 or not np.all(np.isfinite(axis))
        or np.any(np.diff(axis) <= 0)
    ):
        raise ValueError("IQ axis must have at least three finite increasing points")


def generate_singleshot_ge(
    iq_axis: NDArray[np.float64], truth: GaussianCloudParameters, seed: int
) -> NDArray[np.complex128]:
    """Return raw complex IQ with shape (2, shots), rows prepared g then e.

    iq_axis gives discrete sample coordinates for BOTH real and imaginary
    quadratures. Choose a fine grid extending several widths beyond both
    centers in each quadrature; Gaussian tails outside this grid are truncated.
    The public fitting.gauss_func supplies normalized masses on this grid.
    truth supplies both centers, quadrature noise width, per-state shots and
    symmetric state misassignment. seed initializes a local RNG. Inputs are
    unchanged. Invalid grid, width, shot count or probability raise ValueError.
    """
    _iq_axis(iq_axis)
    if not np.isfinite(truth.width) or truth.width <= 0 or truth.shots <= 0:
        raise ValueError("width and shots must be positive")
    if not 0 <= truth.misassignment <= 1:
        raise ValueError("misassignment must be in [0, 1]")
    rng = np.random.default_rng(seed)
    clouds = np.empty((2, truth.shots), dtype=np.complex128)
    for prepared in range(2):
        states = np.full(truth.shots, prepared)
        wrong = rng.random(truth.shots) < truth.misassignment
        states[wrong] = 1 - prepared
        for state, center in enumerate((truth.center_g, truth.center_e)):
            mask = states == state
            count = int(np.count_nonzero(mask))
            real = rng.choice(
                iq_axis, count, p=gauss_func(iq_axis, center.real, truth.width)
            )
            imag = rng.choice(
                iq_axis, count, p=gauss_func(iq_axis, center.imag, truth.width)
            )
            clouds[prepared, mask] = real + 1j * imag
    return clouds


def generate_singleshot_rabi(
    lengths: NDArray[np.float64],
    bin_edges: NDArray[np.float64],
    truth: RabiPhysicalParams,
    shots: int,
    noise_std: float,
    seed: int,
) -> NDArray[np.complex128]:
    """Sample raw complex IQ with shape (len(lengths), shots).

    lengths is a finite 1D sweep in truth's coordinate units. bin_edges is a
    finite increasing projected-IQ grid. Public model_bin_probabilities supplies
    per-row probabilities including truth.sigma Gaussian projected noise and
    readout transitions; samples use bin midpoints. Use bins much narrower than
    sigma and a range covering both centers and their tails. shots is positive
    per coordinate. noise_std is independent perpendicular (imaginary) Gaussian
    IQ noise; it does NOT add a second noise term along the projection.
    truth's center_g/e are real projected IQ centers. seed initializes a local
    RNG. Inputs are unchanged. Invalid grids, shot count, noise or physical
    parameters raise ValueError.
    """
    _iq_axis(bin_edges)
    if lengths.ndim != 1 or lengths.size == 0 or not np.all(np.isfinite(lengths)):
        raise ValueError("lengths must be a finite nonempty 1D array")
    if shots <= 0 or not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("shots must be positive and noise_std nonnegative finite")
    probabilities = model_bin_probabilities(lengths, bin_edges, truth)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("invalid physical parameters for supplied lengths")
    rng = np.random.default_rng(seed)
    centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    signals = np.empty((lengths.size, shots), dtype=np.complex128)
    for index, masses in enumerate(probabilities):
        real = rng.choice(centers, shots, p=masses)
        signals[index] = real + 1j * rng.normal(0, noise_std, shots)
    return signals
