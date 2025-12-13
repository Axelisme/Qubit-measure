import numpy as np


def van_der_corput(n, base=2):
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


def vdc_permutation(n, base=2):
    """
    Generate a permutation of n elements based on the van der Corput sequence.
    """
    vdc_sequence = van_der_corput(n, base)
    return np.argsort(vdc_sequence)
