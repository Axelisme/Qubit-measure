"""Shared continuous-flux Hamiltonian evaluation for adiabatic and ODE kernels."""

from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

TERM_KINDS = ("cos_2pi", "sin_2pi", "cos_pi", "sin_pi")


class PackedExpansion(NamedTuple):
    """Fixed-coordinate H/h in GHz, packed by each consuming kernel.

    static is the square constant matrix. matrices holds one same-size matrix
    per term. loops holds each term's waveform column; kinds indexes TERM_KINDS.
    Arrays are borrowed without mutation; their term order sets summation order.
    """

    static: NDArray[np.complex128]
    matrices: NDArray[np.complex128]
    loops: NDArray[np.int64]
    kinds: NDArray[np.int64]


@njit(nogil=True, cache=True)
def flux_hamiltonian(
    expansion: PackedExpansion,
    times: NDArray[np.float64],
    flux: NDArray[np.float64],
    time: float,
) -> NDArray[np.complex128]:
    """Return fixed-coordinate H/h in GHz at absolute time in ns.

    expansion contains aligned term arrays with valid kind codes and loop
    columns. times is a nonempty increasing sample axis in ns; flux has one
    row per sample and absolute reduced-flux columns. Interpolation is linear
    with endpoint hold. Inputs are borrowed, and the returned matrix is new.
    Packing and waveform validation belong to the consuming kernel.
    """
    h = expansion.static.copy()
    for term in range(len(expansion.loops)):
        loop = expansion.loops[term]
        value = np.interp(time, times, flux[:, loop])
        kind = expansion.kinds[term]
        angle = np.pi * value * (2 if kind < 2 else 1)
        coefficient = np.cos(angle) if kind % 2 == 0 else np.sin(angle)
        h += coefficient * expansion.matrices[term]
    return h
