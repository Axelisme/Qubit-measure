"""Shared complex envelope interpolation for Python and numba consumers."""

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(nogil=True, cache=True)
def sample_value(
    local_ns: float,
    sample_ns: NDArray[np.float64],
    samples: NDArray[np.complex128],
) -> complex:
    """Evaluate one pulse-local time using linear real/imaginary interpolation.

    Empty samples denote the unity envelope. Otherwise sample_ns and samples
    are aligned vectors with increasing times beginning at zero. The first and
    last samples are held outside the grid. Callers own finite/nonnegative
    time validation and pulse support; this routine performs no validation.
    """
    if len(samples) == 0:
        return 1 + 0j
    return complex(
        np.interp(local_ns, sample_ns, samples.real),
        np.interp(local_ns, sample_ns, samples.imag),
    )


@njit(nogil=True, cache=True)
def sample_values(
    local_ns: NDArray[np.float64],
    sample_ns: NDArray[np.float64],
    samples: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Evaluate an array of pulse-local ns, preserving its shape.

    Inputs follow sample_value's preconditions; an empty grid denotes unity.
    The returned complex128 array is owned by the caller.
    """
    times = local_ns.ravel()
    result = np.empty(times.size, dtype=np.complex128)
    for index in range(times.size):
        result[index] = sample_value(times[index], sample_ns, samples)
    return result.reshape(local_ns.shape)
