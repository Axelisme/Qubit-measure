"""Raw-IQ sampling from the public singleshot fitting distributions."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fitting import gauss_func


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
