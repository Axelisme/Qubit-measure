"""Stateless numpy numeric scalar projection for persistence boundaries."""

import numpy as np


def normalize_numpy_scalar(value: object) -> object:
    """Project one numpy floating/integer scalar to Python float/int.

    Floating scalars use normal Python float rounding; integer scalars retain
    their exact value. Other inputs are returned unchanged, without recursion
    or container copying. This function does not validate finite values or
    strict JSON: the consuming boundary must reject non-finite results,
    including finite extended floats that overflow Python float.
    """
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
