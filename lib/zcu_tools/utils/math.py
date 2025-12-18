import numpy as np
from numpy.typing import NDArray


def van_der_corput(n: int, base: int = 2) -> NDArray[np.float64]:
    """
    Generate n elements of a van der Corput sequence in base 'base'.
    """
    # Indices
    k = np.arange(n)
    vdc = np.zeros(n, dtype=float)
    denom = 1.0

    while np.any(k > 0):
        k, remainder = divmod(k, base)
        denom *= base
        vdc += remainder / denom

    return vdc


def vdc_permutation(n: int, base: int = 2) -> NDArray[np.int64]:
    """
    Generate a permutation of n elements based on the van der Corput sequence.
    """
    vdc_sequence = van_der_corput(n, base)
    return np.argsort(vdc_sequence)
