"""Axis operations on herald Qubit1/Qubit2 × terminal Qubit1/Qubit2 counts."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from zcu_lab.v2.support.paired_counts import joint_probabilities


def terminal_counts(
    sums: NDArray[np.float64],
    herald: tuple[Literal["g", "e", "other"], Literal["g", "e", "other"]],
) -> NDArray[np.float64]:
    """Select terminal (...,3,3) counts for the named herald (Qubit1,Qubit2) classes.

    sums ends in herald Qubit1/Qubit2 then terminal Qubit1/Qubit2, each g/e/other. Leading
    scan dimensions remain unchanged. This returns a view without normalization.
    Invalid class names raise ValueError. Unmeasured NaNs propagate.
    """
    qubit1_class, qubit2_class = (("g", "e", "other").index(state) for state in herald)
    return sums[..., qubit1_class, qubit2_class, :, :]


def herald_probabilities(sums: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return (..., 3, 3) herald probabilities from 81 counts.

    Sum over terminal Qubit1/Qubit2 first. The returned (g,g) cell is pass fraction.
    """
    return joint_probabilities(sums.sum(axis=(-2, -1)))
