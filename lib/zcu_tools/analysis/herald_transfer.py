"""Self-calibration of shared herald and signal IQ readout."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TransferResult:
    """Self-calibrated readout and state transfer, ordered by population.

    joint_probabilities: (N, n, n) joint herald/signal probabilities J.
    transition_matrices: (N, n, n) row-normalized K[i, j] = P(i -> j).
    change_probability: (N,) probability C = 1 - trace(J).
    populations: (n,) mean herald populations, in descending order.
    centers: (n,) complex state centers in the original IQ units.
    sigma: Gaussian standard deviation per quadrature, in original IQ units.
    singular_values: (n+1,) subspace singular values divided by the largest.
    Unmeasured points have NaN entries in J, K and C.
    """

    joint_probabilities: NDArray[np.float64]
    transition_matrices: NDArray[np.float64]
    change_probability: NDArray[np.float64]
    populations: NDArray[np.float64]
    centers: NDArray[np.complex128]
    sigma: float
    singular_values: NDArray[np.float64]


def selfcal_transfer(
    sums: NDArray[np.float64],
    n_states: int = 2,
    *,
    previous: TransferResult | None = None,
) -> TransferResult:
    """Infer shared readout calibration and transfer from (N, 6, 6) IQ sums."""
    raise NotImplementedError("Shared-readout self-calibration is not implemented")
