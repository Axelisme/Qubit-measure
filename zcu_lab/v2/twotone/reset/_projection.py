from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from zcu_tools.utils.process import find_rotate_angle


def principal_iq_direction(reference: NDArray[np.complex128]) -> tuple[float, float]:
    """Return the I/Q weights of the principal axis through the supplied IQ means.

    Pass the branch or branches used to estimate the common axis. Missing means
    are ignored. Use the I axis until there are at least two observed means.
    """
    observed = reference[np.isfinite(reference)]
    if observed.size < 2:
        return (1.0, 0.0)
    angle = find_rotate_angle(observed)
    return (float(np.cos(angle)), float(np.sin(angle)))
