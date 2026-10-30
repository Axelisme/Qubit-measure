"""Shared validation at the acquisition and herald inner-processor boundaries."""

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


def validated_payload(payload: object, *, shots: int) -> NDArray[np.float64]:
    """Return a processor payload of shape (shots, out_reads, N).

    Raise ValueError for non-ndarray/non-float64 values, rank other than three,
    a mismatched shot count, or empty output axes. NaN and infinity are allowed.
    """
    if not isinstance(payload, np.ndarray):
        raise ValueError("processor must return an ndarray")
    if payload.dtype != np.float64:
        raise ValueError("processor output dtype must be float64")
    if payload.ndim != 3:
        raise ValueError("processor output must have rank three")
    if payload.shape[0] != shots:
        raise ValueError(f"processor output shots must equal {shots}")
    if payload.shape[1] == 0 or payload.shape[2] == 0:
        raise ValueError("processor output axes must be nonempty")
    return payload
