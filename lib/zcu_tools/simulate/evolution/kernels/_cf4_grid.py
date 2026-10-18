"""CF4 spectral-width and nested-grid policy shared by builders and estimates."""

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def spectral_width(energies, x, y, gain=1.0):
    """Return bare span plus twice the gain-weighted quadrature norms, in GHz."""
    return (
        np.max(energies.real)
        - np.min(energies.real)
        + 2 * abs(gain) * (np.linalg.norm(x) + np.linalg.norm(y))
    )


@njit(nogil=True, cache=True)
def coarse_count(scaled_span):
    """Round a spectral-turn estimate up to at least eight CF4 intervals.

    Callers supply the 1.5 safety factor with their existing arithmetic order.
    """
    return max(8, 8 * int(np.ceil(scaled_span / 8)))


@njit(nogil=True, cache=True)
def finer_count(count):
    """Return the next 1.5-times grid, keeping eight shared probe intervals."""
    return 8 * int(np.ceil(1.5 * count / 8))
