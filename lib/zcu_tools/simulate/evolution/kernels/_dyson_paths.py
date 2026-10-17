"""Preallocated contraction of the complete shared modal Dyson path table."""

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(nogil=True, cache=True)
def _shared_path_capacity(
    modes: NDArray[np.complex128], couplings: NDArray[np.complex128]
) -> int:
    """Count nonzero coupling/input combinations before coefficient underflow."""
    dimension = len(modes)
    inner_counts = np.zeros(dimension, dtype=np.int64)
    for b in range(dimension):
        for inner in range(2):
            for c in range(dimension):
                if couplings[inner, b, c] != 0 and modes[c] != 0:
                    inner_counts[b] += 1
    capacity = 0
    for a in range(dimension):
        for outer in range(2):
            for b in range(dimension):
                if couplings[outer, a, b] != 0:
                    capacity += int(inner_counts[b]) + int(modes[b] != 0)
    return capacity


@njit(nogil=True, cache=True)
def build_shared_paths(
    modes: NDArray[np.complex128], couplings: NDArray[np.complex128]
) -> tuple[NDArray[np.int64], NDArray[np.complex128]]:
    """Contract every nonzero first- and second-order path with one modal input.

    modes is the common density's finite complex modal vector (D,), D>0.
    couplings contains its model/port's finite drive matrices (2,D,D), with
    carrier signs -1,+1 in that order. The internal caller owns shape and model
    validation. Inputs are unchanged. Allocation failures propagate MemoryError;
    numerical overflow is checked by the downstream pruning boundary.

    Return new writable indices (P,5) and coefficients (P,). Records are
    (a,b,c,r,s), with c=-1,s=0 for first order and s=-1,+1 for second order.
    They follow a, outer sign, b, first order, inner sign, c enumeration order.
    Coefficients include the input modes; products that underflow to zero vanish.

    Shared pruning needs zero coefficient cutoffs, so gain, duration and the
    budget cannot remove a nonzero candidate here. Count an upper bound before
    filling arrays to avoid a large intermediate list of five-int tuples.
    Products that underflow to zero leave unused capacity at the end.
    """
    dimension = len(modes)
    capacity = _shared_path_capacity(modes, couplings)
    indices = np.empty((capacity, 5), dtype=np.int64)
    coefficients = np.empty(capacity, dtype=np.complex128)
    count = 0
    for a in range(dimension):
        for outer in range(2):
            r = 2 * outer - 1
            for b in range(dimension):
                coefficient = couplings[outer, a, b]
                if coefficient == 0:
                    continue
                first = coefficient * modes[b]
                if abs(first) > 0:
                    indices[count] = (a, b, -1, r, 0)
                    coefficients[count] = first
                    count += 1
                for inner in range(2):
                    s = 2 * inner - 1
                    for c in range(dimension):
                        second = coefficient * couplings[inner, b, c] * modes[c]
                        if abs(second) > 0:
                            indices[count] = (a, b, c, r, s)
                            coefficients[count] = second
                            count += 1
    return indices[:count], coefficients[:count]


def contractible_modes(
    modes: NDArray[np.complex128],
    couplings: NDArray[np.complex128],
    weights: NDArray[np.float64],
    conjugates: NDArray[np.int64],
    *,
    strength: float,
    allowance: float,
) -> tuple[NDArray[np.complex128], float]:
    """Omit conjugate input-mode groups using a bound on all their driven paths.

    weights[a] bounds each reconstructed density entry of output mode a.
    strength is max gain times max duration. The caller includes free modal
    exponential growth in weights. Summing absolute couplings bounds every
    first/second-order path from a source mode without enumerating those paths. Return a copy for
    driven-path construction and the spent entrywise bound. Free propagation
    must still use the original modes. Inputs are unchanged.
    """
    absolute = np.abs(couplings).sum(axis=0)
    first = weights @ absolute
    bounds = abs(modes) * (strength * first + strength**2 / 2 * (first @ absolute))
    representatives = np.flatnonzero(np.arange(len(modes)) <= conjugates)
    group_bounds = bounds[representatives] + np.where(
        representatives == conjugates[representatives],
        0.0,
        bounds[conjugates[representatives]],
    )
    contracted = modes.copy()
    spent = 0.0
    for index in np.argsort(group_bounds, kind="stable"):
        total = spent + float(group_bounds[index])
        if total > allowance:
            break
        mode = representatives[index]
        contracted[mode] = contracted[conjugates[mode]] = 0
        spent = total
    return contracted, spent
