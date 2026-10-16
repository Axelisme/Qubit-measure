"""Closed-system integer propagator composition."""

from operator import index

import numpy as np

from .state import ComplexMatrix, owned_matrix


def propagator_power(propagator: ComplexMatrix, repetitions: int) -> ComplexMatrix:
    """Return U**N by binary matrix multiplication; N=0 returns identity.

    U is a finite square complex matrix, copied, never modified. N must be a
    nonnegative Python/NumPy integer (not bool), otherwise ValueError. This tool
    composes identical closed-system propagators, not dissipative evolution.
    It does not certify unitarity or normalize the result.
    """
    try:
        n = index(repetitions)
    except TypeError as error:
        raise ValueError("repetitions must be a nonnegative integer") from error
    if isinstance(repetitions, (bool, np.bool_)) or n < 0:
        raise ValueError("repetitions must be a nonnegative integer")
    base = owned_matrix(propagator, "propagator").copy()
    result = np.eye(base.shape[0], dtype=np.complex128)
    while n:
        if n & 1:
            result = result @ base
        n >>= 1
        if n:
            base = base @ base
    if not np.isfinite(result).all():
        raise RuntimeError("propagator power produced nonfinite values")
    return result
