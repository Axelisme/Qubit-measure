"""Validation at the joint-ADC shot-sum processor boundary."""

import numpy as np
from numpy.typing import NDArray


def validated_sums(payload: object) -> NDArray[np.float64]:
    """Require float64 ndarray per-point sums, reporting expected and actual dtype."""
    if not isinstance(payload, np.ndarray) or payload.dtype != np.float64:
        raise ValueError(
            f"processor expected float64 ndarray; actual {type(payload).__name__} "
            f"dtype={getattr(payload, 'dtype', None)}"
        )
    return payload
